use openai_dive::v1::resources::chat::ChatCompletionChunkResponse;
use qwen3_deploy::chat_stream;
use rocket::Request;
use rocket::futures::{Stream, StreamExt};
use rocket::http::ContentType;
use rocket::response::Responder;
use rocket::response::stream::TextStream;
use rocket::serde::json::Json;

enum Response<R: Stream<Item = String> + Send> {
    Stream(TextStream<R>),
    Text(String),
}

impl<'r, 'o: 'r, R> Responder<'r, 'o> for Response<R>
where
    R: Stream<Item = String> + Send + 'o,
    'r: 'o,
{
    fn respond_to(self, req: &'r Request<'_>) -> rocket::response::Result<'o> {
        match self {
            Response::Stream(stream) => stream.respond_to(req),
            Response::Text(text) => text.respond_to(req),
        }
    }
}
#[post("/completions", data = "<req>")]
pub(crate) async fn chat(req: String) -> (ContentType, Response<impl Stream<Item = String>>) {
    let stream = TextStream! {
        match chat_stream(&req) {
            Ok(stream) => {
                let mut boxed_stream = Box::pin(stream);
                while let Some(resp) = boxed_stream.next().await {
                    yield format!("event: message\ndata: {}\n\n", resp);
                }
                yield format!("event: message\ndata: {}\n\n", "[DONE]");
            }
            Err(e) => {
                yield format!("event: error\ndata: {}\n\n", e.to_string());
            }
        }
    };
    (ContentType::EventStream, Response::Stream(stream))
}
