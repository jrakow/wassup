use criterion::black_box;
use parity_wasm::elements::ValueType;
use serde_json::json;
use std::time::Instant;
use wassup::{improve_snippet, SuperoptResult, ValueTypeConfig};
use wassup_encoding::{Instruction::*, Value::*};

fn main() {
	let timeout_ms = 60 * 60 * 1000;

	let mut results: serde_json::Value =
		serde_json::from_reader(std::fs::File::open("bench_results.json").unwrap()).unwrap();

	for bench_name in &[
		"encoding_size",
		"initial_stack_values",
		"length",
		"local_variables",
	] {
		let bench = match *bench_name {
			"encoding_size" => encoding_size,
			"initial_stack_values" => initial_stack_values,
			"length" => length,
			"local_variables" => local_variables,
			_ => unreachable!(),
		};

		let results = &mut results[bench_name];
		for i64_enabled in &[false, true] {
			for n in 1..32 {
				eprint!("({}, {}, {}): ", bench_name, n, i64_enabled);

				let res = bench(*i64_enabled, n, timeout_ms);

				let diff = if let Some(diff) = res {
					diff
				} else {
					eprintln!("timed out");
					break;
				};

				let mins = diff.as_secs() / 60;
				let secs = diff.as_secs() % 60;
				let millis = diff.subsec_millis();

				if mins > 0 {
					eprint!("{}min ", mins);
				}
				if secs > 0 {
					eprint!("{}s ", secs);
				}
				eprintln!("{}ms ", millis);

				if !i64_enabled {
					results["i32"][format!("{}", n)] = json!(diff.as_millis() as usize);
				} else {
					results["mixed"][format!("{}", n)] = json!(diff.as_millis() as usize);
				}
			}
		}
	}

	serde_json::to_writer(
		std::fs::File::create("bench_results.json").unwrap(),
		&results,
	)
	.unwrap();
}

fn encoding_size(
	i64_enabled: bool,
	i32_size: usize,
	timeout_ms: usize,
) -> Option<std::time::Duration> {
	let source = &[Const(I32(0)), I32Add];

	let start = Instant::now();
	let res = improve_snippet(
		black_box(source),
		&[],
		ValueTypeConfig {
			i32_size,
			i64_size: Some(2 * i32_size).filter(|_| i64_enabled),
		},
		Some(timeout_ms),
	);
	let diff = Instant::now() - start;

	Some(diff).filter(|_| res != SuperoptResult::Timeout)
}

fn initial_stack_values(
	i64_enabled: bool,
	initial_stack_size: usize,
	timeout_ms: usize,
) -> Option<std::time::Duration> {
	let source = if initial_stack_size == 1 {
		vec![I32Eqz]
	} else {
		vec![I32Eq; initial_stack_size - 1]
	};

	let start = Instant::now();
	let res = improve_snippet(
		black_box(&source),
		&[],
		ValueTypeConfig {
			i32_size: 4,
			i64_size: Some(8).filter(|_| i64_enabled),
		},
		Some(timeout_ms),
	);
	let diff = Instant::now() - start;

	Some(diff).filter(|_| res != SuperoptResult::Timeout)
}

fn length(i64_enabled: bool, length: usize, timeout_ms: usize) -> Option<std::time::Duration> {
	let source = if length == 1 {
		vec![Nop]
	} else {
		let mut v = vec![Const(I32(0))];
		v.resize(length - 1, Nop);
		v.push(I32Add);
		v
	};
	assert_eq!(source.len(), length);

	let start = Instant::now();
	let res = improve_snippet(
		black_box(&source),
		&[],
		ValueTypeConfig {
			i32_size: 4,
			i64_size: Some(8).filter(|_| i64_enabled),
		},
		Some(timeout_ms),
	);
	let diff = Instant::now() - start;

	Some(diff).filter(|_| res != SuperoptResult::Timeout)
}

fn local_variables(
	i64_enabled: bool,
	n_locals: usize,
	timeout_ms: usize,
) -> Option<std::time::Duration> {
	let source: Vec<_> = (0..n_locals).map(|i| TeeLocal(i as u32)).collect();

	let start = Instant::now();
	let res = improve_snippet(
		black_box(&source),
		&vec![ValueType::I32; n_locals],
		ValueTypeConfig {
			i32_size: 4,
			i64_size: Some(8).filter(|_| i64_enabled),
		},
		Some(timeout_ms),
	);
	let diff = Instant::now() - start;

	Some(diff).filter(|_| res != SuperoptResult::Timeout)
}
