#![feature(nll)]

mod lexer;
mod parser;
mod tyck;
mod util;

use std::{fs::File, io::Read};

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

    let program = match parse_file(&bump, file_id, &buf) {
        Ok(program) => program,
        Err(lalrpop_util::ParseError::User { error }) => {
            let diagnostic = error.into_diagnostic();
            let writer = StandardStream::stderr(ColorChoice::Always);
            let config = codespan_reporting::term::Config::default();
            term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
            std::process::exit(1);
        }
        Err(err) => {
            eprintln!("{:?}", err);
            std::process::exit(1);
        }
    };

    match tyck::tyck_program(&program) {
        Ok(_) => (),
        Err(error) => {
            let diagnostic = error.into_diagnostic();
            let writer = StandardStream::stderr(ColorChoice::Always);
            let mut config = codespan_reporting::term::Config::default();
            config.start_context_lines = 10;
            config.end_context_lines = 10;
            term::emit(&mut writer.lock(), &config, &files, &diagnostic).unwrap();
            std::process::exit(1);
        }
    }
}
