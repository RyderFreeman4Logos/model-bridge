use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_core::Stream;

/// Reassembles raw byte chunks into complete SSE data lines.
///
/// Wraps an inner byte stream and yields complete `data: ` payloads
/// (with the `data: ` prefix stripped) one line at a time.
pub struct SseLineParser<S> {
    inner: Pin<Box<S>>,
    buffer: String,
}

impl<S> SseLineParser<S> {
    pub fn new(inner: S) -> Self {
        Self {
            inner: Box::pin(inner),
            buffer: String::new(),
        }
    }
}

impl<S, E> Stream for SseLineParser<S>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    type Item = Result<String, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();

        loop {
            // Try to extract a complete line from the buffer first.
            if let Some(line) = take_next_data_line(&mut this.buffer) {
                return Poll::Ready(Some(Ok(line)));
            }

            // Need more data from the inner stream.
            match this.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(Ok(bytes))) => {
                    // Append valid UTF-8 from the chunk, skip malformed bytes.
                    match std::str::from_utf8(&bytes) {
                        Ok(s) => this.buffer.push_str(s),
                        Err(_) => {
                            // Best-effort: lossy conversion skips invalid sequences.
                            this.buffer.push_str(&String::from_utf8_lossy(&bytes));
                        }
                    }
                    // Loop to try extracting a line from the extended buffer.
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => {
                    // Stream ended. Drain any remaining data line in buffer.
                    if let Some(line) = take_next_data_line(&mut this.buffer) {
                        return Poll::Ready(Some(Ok(line)));
                    }
                    return Poll::Ready(None);
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Scans the buffer for the next complete line that starts with `data: `.
/// Consumes all lines up to and including the matching one.
/// Skips empty lines and SSE comments (lines starting with `:`).
fn take_next_data_line(buffer: &mut String) -> Option<String> {
    loop {
        let newline_pos = buffer.find('\n')?;
        let line = buffer[..newline_pos].trim_end_matches('\r').to_owned();

        // Consume the line plus the newline character.
        buffer.drain(..=newline_pos);

        // Skip empty lines (keep-alive) and comment lines.
        if line.is_empty() || line.starts_with(':') {
            continue;
        }

        // Return the payload for data lines.
        if let Some(payload) = line.strip_prefix("data: ") {
            return Some(payload.to_owned());
        }

        // For non-data SSE fields (event:, id:, retry:), skip.
        // Also return raw lines that don't match any SSE field as-is
        // (Ollama sends raw JSON without SSE prefix).
        return Some(line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use std::pin::Pin;
    use std::task::{Context, Poll, Waker};

    /// Helper stream that yields pre-defined byte chunks.
    struct MockByteStream {
        chunks: Vec<Bytes>,
        index: usize,
    }

    impl MockByteStream {
        fn new(chunks: Vec<&str>) -> Self {
            Self {
                chunks: chunks
                    .into_iter()
                    .map(|s| Bytes::from(s.to_owned()))
                    .collect(),
                index: 0,
            }
        }
    }

    impl Stream for MockByteStream {
        type Item = Result<Bytes, Infallible>;

        fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();
            if this.index < this.chunks.len() {
                let chunk = this.chunks[this.index].clone();
                this.index += 1;
                Poll::Ready(Some(Ok(chunk)))
            } else {
                Poll::Ready(None)
            }
        }
    }

    /// Collect all items from an SseLineParser synchronously (works because
    /// MockByteStream always returns Ready).
    fn collect_lines(parser: &mut SseLineParser<MockByteStream>) -> Vec<String> {
        let mut results = Vec::new();
        let waker = Waker::noop();
        let mut cx = Context::from_waker(&waker);

        loop {
            match Pin::new(&mut *parser).poll_next(&mut cx) {
                Poll::Ready(Some(Ok(line))) => results.push(line),
                Poll::Ready(Some(Err(_))) => unreachable!(),
                Poll::Ready(None) => break,
                Poll::Pending => break,
            }
        }
        results
    }

    #[test]
    fn test_single_complete_event() {
        let stream = MockByteStream::new(vec!["data: {\"hello\":\"world\"}\n\n"]);
        let mut parser = SseLineParser::new(stream);
        let lines = collect_lines(&mut parser);
        assert_eq!(lines, vec!["{\"hello\":\"world\"}"]);
    }

    #[test]
    fn test_multi_chunk_reassembly() {
        let stream = MockByteStream::new(vec!["data: {\"he", "llo\":\"world\"}\n\n"]);
        let mut parser = SseLineParser::new(stream);
        let lines = collect_lines(&mut parser);
        assert_eq!(lines, vec!["{\"hello\":\"world\"}"]);
    }

    #[test]
    fn test_keepalive_skipped() {
        let stream = MockByteStream::new(vec!["\n\n\ndata: ok\n\n"]);
        let mut parser = SseLineParser::new(stream);
        let lines = collect_lines(&mut parser);
        assert_eq!(lines, vec!["ok"]);
    }

    #[test]
    fn test_comment_lines_skipped() {
        let stream = MockByteStream::new(vec![": this is a comment\ndata: payload\n\n"]);
        let mut parser = SseLineParser::new(stream);
        let lines = collect_lines(&mut parser);
        assert_eq!(lines, vec!["payload"]);
    }

    #[test]
    fn test_multiple_events_in_one_chunk() {
        let stream = MockByteStream::new(vec!["data: first\ndata: second\n\n"]);
        let mut parser = SseLineParser::new(stream);
        let lines = collect_lines(&mut parser);
        assert_eq!(lines, vec!["first", "second"]);
    }

    #[test]
    fn test_done_sentinel_passed_through() {
        let stream = MockByteStream::new(vec!["data: [DONE]\n\n"]);
        let mut parser = SseLineParser::new(stream);
        let lines = collect_lines(&mut parser);
        assert_eq!(lines, vec!["[DONE]"]);
    }

    #[test]
    fn test_raw_json_lines_passed_through() {
        // Ollama streams raw JSON without SSE prefix.
        let stream = MockByteStream::new(vec![
            "{\"message\":{\"content\":\"Hi\"}}\n{\"done\":true}\n",
        ]);
        let mut parser = SseLineParser::new(stream);
        let lines = collect_lines(&mut parser);
        assert_eq!(
            lines,
            vec!["{\"message\":{\"content\":\"Hi\"}}", "{\"done\":true}"]
        );
    }
}
