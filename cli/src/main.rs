use clap::{crate_authors, crate_description, crate_version, App, Arg, ArgMatches};
use parity_wasm::{self, deserialize_buffer, elements::Module, serialize};
use std::{
	ffi::OsStr,
	fs::File,
	io::{stdin, stdout, Read, Write},
	path::Path,
	process::exit,
	str::FromStr,
};
use wabt::wat2wasm;

fn main() {
	let args = App::new("wassup")
		.version(crate_version!())
		.author(crate_authors!("\n"))
		.about(crate_description!())
		.arg(
			Arg::with_name("STDIN")
				.long("--stdin")
				.help("Read the input from stdin"),
		)
		.arg(
			Arg::with_name("INPUT")
				.help("Input file to optimize")
				.required_unless("STDIN")
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
		.arg(
			Arg::with_name("OPT_SIZE_I32")
				.long("opt-size-i32")
				.value_name("SIZE")
				.help("Size of a 32 bit integer in optimization")
				.number_of_values(1)
				.default_value("4"),
		)
		.arg(
			Arg::with_name("OPT_SIZE_I64")
				.long("opt-size-i64")
				.value_name("SIZE")
				.help("Size of a 64 bit integer in optimization, 0 if i64 disabled")
				.number_of_values(1)
				.default_value("8"),
		)
		.arg(
			Arg::with_name("TRANSVAL_SIZE_I32")
				.long("transval-size-i32")
				.value_name("SIZE")
				.help("Size of a 32 bit integer in translation validation")
				.number_of_values(1)
				.default_value("6"),
		)
		.arg(
			Arg::with_name("TRANSVAL_SIZE_I64")
				.long("transval-size-i64")
				.value_name("SIZE")
				.help("Size of a 64 bit integer in translation validation, 0 if i64 disabled")
				.number_of_values(1)
				.default_value("12"),
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
		let mut input_file: Box<dyn Read> = if let Some(path) = args.value_of("INPUT") {
			Box::new(File::open(path).map_err(|e: std::io::Error| {
				format!("Could not open input file: {}", e.to_string())
			})?)
		} else {
			Box::new(stdin())
		};

		let mut buffer = Vec::new();
		input_file
			.read_to_end(&mut buffer)
			.map_err(|e: std::io::Error| format!("Could not read input file: {}", e.to_string()))?;

		if &buffer[0..4] != b"\0asm" {
			buffer =
				wat2wasm(&buffer).map_err(|_| "Could not parse input as WAT module".to_string())?;
		}

		deserialize_buffer(&buffer).map_err(|e: parity_wasm::elements::Error| {
			format!("Could not parse input as WASM module: {}", e.to_string())
		})?
	};

	let opt_value_type_config = wassup::ValueTypeConfig {
		i32_size: usize::from_str(args.value_of("OPT_SIZE_I32").unwrap())
			.map_err(|_| "Could not parse OPT_SIZE_I32 as a size")?,
		i64_size: match args.value_of("OPT_SIZE_I64").unwrap() {
			"0" => None,
			x @ _ => {
				Some(usize::from_str(x).map_err(|_| "Could not parse OPT_SIZE_I64 as a size")?)
			}
		},
	};

	let transval_value_type_config = {
		let i32_size = usize::from_str(args.value_of("TRANSVAL_SIZE_I32").unwrap())
			.map_err(|_| "Could not parse TRANSVAL_SIZE_I32 as a size")?;
		let i64_size = usize::from_str(args.value_of("TRANSVAL_SIZE_I64").unwrap())
			.map_err(|_| "Could not parse TRANSVAL_SIZE_I64 as a size")?;

		match (i32_size, i64_size) {
			(0, 0) => None,
			(_, 0) => Some(wassup::ValueTypeConfig {
				i32_size,
				i64_size: None,
			}),
			(0, _) => Err("TRANSVAL_SIZE_I32 needs to be enabled to enable TRANSVAL_SIZE_I64")?,
			(_, _) => Some(wassup::ValueTypeConfig {
				i32_size,
				i64_size: Some(i64_size),
			}),
		}
	};

	wassup::superoptimize_module(
		&mut input_module,
		opt_value_type_config,
		transval_value_type_config,
	);

	let mut buffer = serialize(input_module).unwrap();

	let output_path = args.value_of("OUTPUT");

	// detect output format
	// wat iff stdout or (specified and ends in "wat")
	if output_path == None || Path::new(output_path.unwrap()).extension() == Some(OsStr::new("wat"))
	{
		// wat
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
