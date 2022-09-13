pub mod app;
pub mod utils;
use app::*;

fn main() {
    let mut app = App::new();
    app.run();
}
