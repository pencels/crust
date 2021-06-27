mod lexer;
mod lowering;
mod parser;
mod util;

use std::{fs::File, io::Read};

use bumpalo::Bump;
use codespan_reporting::files::SimpleFiles;
use parser::grammar::ProgramParser;

fn main() {
    let path = "examples/test.crust";
    let mut file = File::open(path).unwrap();
    let mut buf = String::new();
    file.read_to_string(&mut buf).unwrap();

    let bump = Bump::new();
    let mut files: SimpleFiles<String, String> = SimpleFiles::new();

    let file_id = files.add(path.to_string(), buf.clone());

    let program = ProgramParser::new().parse(&bump, file_id, &buf).unwrap();

    for defn in program {
        println!("{:?}", defn);
    }
}
