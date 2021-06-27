mod lexer;
mod lowering;
mod parser;
mod util;

use std::{fs::File, io::Read, ops::Add};

use bumpalo::Bump;
use codespan_derive::IntoDiagnostic;
use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use lalrpop_util::{self, lexer::Token};
use parser::result::ParseError;
use parser::{ast::Defn, grammar::ProgramParser, result};
use util::FileId;

fn parse_file<'a>(
    bump: &'a Bump,
    file_id: FileId,
    source: &'a str,
) -> Result<Vec<Defn<'a>>, lalrpop_util::ParseError<usize, Token<'a>, ParseError>> {
    ProgramParser::new()
        .parse(&bump, file_id, &source)
        .map_err(|error| result::into_crust_error(error, file_id))
}

fn main() {
    let path = "examples/test.crust";
    let mut file = File::open(path).unwrap();
    let mut buf = String::new();
    file.read_to_string(&mut buf).unwrap();

    let bump = Bump::new();
    let mut files: SimpleFiles<String, String> = SimpleFiles::new();

    let file_id = files.add(path.to_string(), buf.clone());

    match parse_file(&bump, file_id, &buf) {
        Ok(program) => {
            for defn in program {
                println!("{:?}", defn);
            }
        }
        Err(lalrpop_util::ParseError::User { error }) => {
            let diagnostic = error.into_diagnostic();
            let writer = StandardStream::stderr(ColorChoice::Always);
            let config = codespan_reporting::term::Config::default();
            term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
        }
        Err(err) => eprintln!("{:?}", err),
    }
}
