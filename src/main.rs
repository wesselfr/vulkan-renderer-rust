pub mod app;
pub mod utils;
use app::*;

fn main() {
    let app = App::new();
    app.run();
}
