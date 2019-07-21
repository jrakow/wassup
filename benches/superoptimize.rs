use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wassup::superoptimize_snippet;
use wassup_encoding::{Instruction::*, Value::*, ValueTypeConfig};

fn const0_add(c: &mut Criterion) {
	c.bench_function("const0_add", |b| {
		b.iter(|| {
			let source = &[Const(I32(0)), I32Add];

			superoptimize_snippet(
				black_box(source),
				&[],
				ValueTypeConfig {
					i32_size: 32,
					i64_size: Some(64),
				},
			)
		})
	});
}

fn consts_add(c: &mut Criterion) {
	c.bench_function("consts_add", |b| {
		b.iter(|| {
			let source = &[Const(I32(1)), Const(I32(2)), I32Add];

			superoptimize_snippet(
				black_box(source),
				&[],
				ValueTypeConfig {
					i32_size: 32,
					i64_size: Some(64),
				},
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
					i32_size: 32,
					i64_size: Some(64),
				},
			)
		})
	});
}

criterion_group! {
	name = benches;
	config = Criterion::default().sample_size(20);
	targets = const0_add, consts_add, const_nop
}
criterion_main!(benches);
