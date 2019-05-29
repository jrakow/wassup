use clap::{crate_authors, crate_description, crate_version, App, Arg, ArgMatches};
use std::{
	ffi::OsStr,
	fs::File,
	io::{stdout, Read, Write},
	path::Path,
	process::exit,
};
use wabt::wat2wasm;
use wassup::parity_wasm::{self, deserialize_buffer, elements::Module, serialize};

fn main() {
	let args = App::new("wassup")
		.version(crate_version!())
		.author(crate_authors!("\n"))
		.about(crate_description!())
		.arg(
			Arg::with_name("INPUT")
				.help("The input file to optimize")
				.required(true)
				.index(1),
		)
		.arg(
			Arg::with_name("OUTPUT")
				.short("o")
				.long("output")
				.value_name("FILE")
				.help("Where to write the result")
				.takes_value(true),
		)
		.get_matches();

	exit(match rmain(args) {
		Ok(()) => 0,
		Err(e) => {
			eprintln!("{}", e);
			1
		}
	})
}

fn rmain(args: ArgMatches) -> Result<(), String> {
	let mut input_module: Module = {
		let path = args.value_of("INPUT").unwrap();
		let mut input_file = File::open(path)
			.map_err(|e: std::io::Error| format!("Could not open input file: {}", e.to_string()))?;

		let mut buffer = Vec::new();
		input_file
			.read_to_end(&mut buffer)
			.map_err(|e: std::io::Error| format!("Could not read input file: {}", e.to_string()))?;

		if &buffer[0..4] != b"\0wasm" {
			buffer = wat2wasm(&buffer).map_err(|_| "Could not read input as WAST".to_string())?;
		}

		deserialize_buffer(&buffer).map_err(|e: parity_wasm::elements::Error| {
			format!("Could not parse input as WASM module: {}", e.to_string())
		})?
	};

	wassup::superoptimize_module(&mut input_module);
	let mut buffer = serialize(input_module).unwrap();

	let output_path = args.value_of("OUTPUT");

	// detect output format
	// wast iff stdout or (specified and ends in "wast")
	if output_path == None
		|| Path::new(output_path.unwrap()).extension() == Some(OsStr::new("wast"))
	{
		// wast
		buffer = wabt::wasm2wat(&buffer).unwrap().as_bytes().to_vec();
	}

	if let Some(path) = output_path {
		let mut file = File::create(path)
			.map_err(|e| format!("Could not open to output file: {}", e.to_string()))?;
		file.write_all(&buffer)
			.map_err(|e| format!("Could not write to output file: {}", e.to_string()))?;
	} else {
		stdout()
			.write_all(&buffer)
			.map_err(|_| "Could not write to stdout")?;
	}

	Ok(())
}
