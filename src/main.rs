#![feature(lazy_cell)]

mod gen;
mod lexer;
mod parser;
mod tyck;
mod util;

use clap::Parser;
use std::{convert::TryFrom, ffi::OsStr, fs::File, io::Read, path::PathBuf, process::Command};

use bumpalo::Bump;
use camino::{Utf8Path, Utf8PathBuf};
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
use tempfile::TempDir;
use util::FileId;

pub type Error = Box<dyn std::error::Error>;

/// Parses a file, producing either a vec of definitions or a parse error.
fn parse_file<'a>(
    bump: &'a Bump,
    file_id: FileId,
    source: &'a str,
) -> Result<Vec<Defn<'a>>, lalrpop_util::ParseError<usize, Token<'a>, ParseError>> {
    ProgramParser::new()
        .parse(&bump, file_id, &source)
        .map_err(|error| result::into_crust_error(error, file_id))
}

/// Loads a file into the file registry, returning its id and contents.
fn load_file(
    files: &mut SimpleFiles<Utf8PathBuf, String>,
    path: &Utf8Path,
) -> Result<(usize, String), Error> {
    let mut file = File::open(path).unwrap();
    let mut buf = String::new();
    file.read_to_string(&mut buf).unwrap();

    let file_id = files.add(Utf8PathBuf::try_from(path.to_path_buf())?, buf.clone());
    Ok((file_id, buf))
}

/// A crusty compiler.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Input file path
    #[arg()]
    input: PathBuf,
    /// Output path
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Output LLVM IR rather than fully compiling
    #[arg(short = 'L')]
    emit_ir_only: bool,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    // Init arena and file source map
    let bump = Bump::new();
    let mut files: SimpleFiles<Utf8PathBuf, String> = SimpleFiles::new();

    // Read input file
    let path = Utf8PathBuf::try_from(args.input)?;
    let (file_id, buf) = load_file(&mut files, &path)?;

    // Parse and handle parser errors.
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

    // Do type checking on the defns and handle the tyck errors.
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

    // Parsing and typechecking done, emit LLVM IR.
    let filename = path.file_stem().unwrap();
    let temp_dir = TempDir::new()?;
    let ll_file_path = if args.emit_ir_only {
        Utf8PathBuf::from(format!("{}.ll", filename))
    } else {
        Utf8PathBuf::try_from(temp_dir.path().to_path_buf())?.join(format!("{}.ll", filename))
    };

    Emitter::emit_ir(
        filename,
        &ll_file_path,
        &program,
        checker
            .struct_tys
            .into_iter()
            .map(|(k, v)| (k, v.expect("ICE: struct should be defined")))
            .collect(),
    )?;

    if args.emit_ir_only {
        return Ok(());
    }

    let ll_obj_file = temp_dir
        .path()
        .join(&format!("{}.o", filename))
        .into_os_string();
    let mut compile_ll_cmd = Command::new("llc-15");
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

    let mut compile_cmd = Command::new("clang-15");
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
