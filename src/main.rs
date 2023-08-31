#![feature(lazy_cell)]

#[macro_use]
extern crate structopt_derive;

mod gen;
mod lexer;
mod parser;
mod tyck;
mod util;

use std::{ffi::OsStr, fs::File, io::Read, path::PathBuf, process::Command};

use bumpalo::Bump;
use codespan_derive::IntoDiagnostic;
use codespan_reporting::{
    files::SimpleFiles,
    term::{
        self,
        termcolor::{ColorChoice, StandardStream},
    },
};
use gen::{Emitter, STDLIB_PATH};
use lalrpop_util::{self, lexer::Token};
use parser::result::ParseError;
use parser::{ast::Defn, grammar::ProgramParser, result};
use structopt::StructOpt;
use tempfile::TempDir;
use util::FileId;

pub type Error = Box<dyn std::error::Error>;

fn parse_file<'a>(
    bump: &'a Bump,
    file_id: FileId,
    source: &'a str,
) -> Result<Vec<Defn<'a>>, lalrpop_util::ParseError<usize, Token<'a>, ParseError>> {
    ProgramParser::new()
        .parse(&bump, file_id, &source)
        .map_err(|error| result::into_crust_error(error, file_id))
}

#[derive(StructOpt, Debug)]
#[structopt(name = "crust", about = "A crusty compiler.")]
struct Opt {
    /// Input file path
    #[structopt(parse(from_os_str))]
    pub input: PathBuf,
    /// Output path
    #[structopt(long = "output", short = "o", parse(from_os_str))]
    pub output: Option<PathBuf>,
}

fn main() -> Result<(), Error> {
    let opt = Opt::from_args();

    let path = opt.input.as_path();
    let mut file = File::open(path).unwrap();
    let mut buf = String::new();
    file.read_to_string(&mut buf).unwrap();

    let bump = Bump::new();
    let mut files: SimpleFiles<String, String> = SimpleFiles::new();

    let file_id = files.add(path.as_os_str().to_string_lossy().into_owned(), buf.clone());

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

    let mut checker = tyck::TypeChecker::new(&bump);
    match checker.tyck_program(&program) {
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

    let filename = path.file_stem().unwrap().to_str().unwrap();
    let temp_dir = TempDir::new()?;

    let ll_file_path = Emitter::emit_ir(
        filename,
        temp_dir.path(),
        opt.output.as_ref().map(|p| p.as_path()),
        &program,
        checker
            .struct_tys
            .into_iter()
            .map(|(k, v)| (k, v.expect("ICE: struct should be defined")))
            .collect(),
    );

    let ll_obj_file = temp_dir
        .path()
        .join(&format!("{}.o", filename))
        .into_os_string();
    let mut compile_ll_cmd = Command::new("llc");
    compile_ll_cmd.args([
        &ll_file_path.as_os_str(),
        OsStr::new("-filetype=obj"),
        OsStr::new("-relocation-model=pic"),
        OsStr::new("-o"),
        &ll_obj_file,
    ]);
    let status = compile_ll_cmd.status().expect("llc command failed to run");
    if !status.success() {
        panic!("aaaa llc returned non-zero exit status");
    }

    let mut compile_cmd = Command::new("clang");
    compile_cmd.args([
        &*STDLIB_PATH,
        &ll_obj_file,
        OsStr::new("-o"),
        OsStr::new(filename),
    ]);
    compile_cmd.status().unwrap();
    let status = compile_cmd.status().expect("clang command failed to run");
    if !status.success() {
        panic!("aaaa clang returned non-zero exit status");
    }

    Ok(())
}
