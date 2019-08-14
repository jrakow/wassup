use criterion::{black_box, criterion_group, criterion_main, Criterion, ParameterizedBenchmark};
use wassup::superoptimize_snippet;
use wassup_encoding::{Instruction::*, Value::*, ValueTypeConfig};

fn const0_add(c: &mut Criterion) {
	c.bench_function_over_inputs(
		"const0_add_i32",
		|b, i32_size: &usize| {
			b.iter(|| {
				let source = &[Const(I32(0)), I32Add];

				superoptimize_snippet(
					black_box(source),
					&[],
					ValueTypeConfig {
						i32_size: *i32_size,
						i64_size: None,
					},
					None,
				)
			})
		},
		1..=32,
	);

	c.bench_function_over_inputs(
		"const0_add_mixed",
		|b, i32_size: &usize| {
			b.iter(|| {
				let source = &[Const(I32(0)), I32Add];

				superoptimize_snippet(
					black_box(source),
					&[],
					ValueTypeConfig {
						i32_size: *i32_size,
						i64_size: Some(*i32_size * 2),
					},
					None,
				)
			})
		},
		1..=32,
	);
}

fn eqz_repeated(c: &mut Criterion) {
	//	c.bench_function_over_inputs(
	//		"eqz_repeated_i32",
	//		|b, n: &usize| {
	//			b.iter(|| {
	//				let mut source = Vec::new();
	//				source.resize(*n, I32Eqz);
	//
	//				superoptimize_snippet(
	//					black_box(&source),
	//					&[],
	//					ValueTypeConfig {
	//						i32_size: 32,
	//						i64_size: None,
	//					},
	//				)
	//			})
	//		},
	//		1..10,
	//	);

	c.bench(
		"eqz_repeated_i32",
		ParameterizedBenchmark::new(
			"eqz_repeated_i32",
			|b, n: &usize| {
				b.iter(|| {
					let mut source = Vec::new();
					source.resize(*n, I32Eqz);

					superoptimize_snippet(
						black_box(&source),
						&[],
						ValueTypeConfig {
							i32_size: 4,
							i64_size: None,
						},
						None,
					)
				})
			},
			1..10,
		)
		.sample_size(2)
		.warm_up_time(std::time::Duration::from_nanos(1)),
	);

	c.bench(
		"eqz_repeated_mixed",
		ParameterizedBenchmark::new(
			"eqz_repeated_mixed",
			|b, n: &usize| {
				b.iter(|| {
					let mut source = Vec::new();
					source.resize(*n, I32Eqz);

					superoptimize_snippet(
						black_box(&source),
						&[],
						ValueTypeConfig {
							i32_size: 4,
							i64_size: Some(8),
						},
						None,
					)
				})
			},
			1..10,
		)
		.sample_size(2)
		.warm_up_time(std::time::Duration::from_nanos(1)),
	);
}

fn consts_add(c: &mut Criterion) {
	c.bench_function("consts_add", |b| {
		b.iter(|| {
			let source = &[Const(I32(1)), Const(I32(2)), I32Add];

			superoptimize_snippet(
				black_box(source),
				&[],
				ValueTypeConfig {
					i32_size: 4,
					i64_size: Some(8),
				},
				None,
			)
		})
	});
}

fn const_nop(c: &mut Criterion) {
	c.bench_function("const_nop", |b| {
		b.iter(|| {
			let source = &[Const(I32(1)), Nop];

			superoptimize_snippet(
				black_box(source),
				&[],
				ValueTypeConfig {
					i32_size: 4,
					i64_size: Some(8),
				},
				None,
			)
		})
	});
}

criterion_group! {
	name = benches;
	config = Criterion::default().sample_size(20);
	targets = const0_add, consts_add, const_nop, eqz_repeated,
}
criterion_main!(benches);
