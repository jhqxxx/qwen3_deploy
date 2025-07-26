use openai_dive::v1::resources::chat::ChatCompletionChunkResponse;
use qwen3_deploy::chat_stream;
use rocket::futures::{Stream, StreamExt};
use rocket::http::ContentType;
use rocket::response::stream::TextStream;
use rocket::serde::json::Json;

#[post("/completions", data = "<req>")]
pub(crate) async fn chat(req: String) -> (ContentType, TextStream<impl Stream<Item = String>>) {
    let stream = TextStream! {
        match chat_stream(&req) {
            Ok(stream) => {
                let mut boxed_stream = Box::pin(stream);
                while let Some(resp) = boxed_stream.next().await {
                    yield format!("event: message\ndata: {}\n\n", resp);
                }
                yield "event: abort\n".to_string()
            }
            Err(_) => {
                yield "Error: Failed to create chat stream".to_string();
            }
        }
    };
    (ContentType::EventStream, stream)
}
