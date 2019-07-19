use criterion::{black_box, criterion_group, criterion_main, Criterion};
use parity_wasm::elements::Instruction::*;
use parity_wasm::elements::ValueType;
use wassup::superoptimize_impl;

fn const0_add(c: &mut Criterion) {
	c.bench_function("const0_add", |b| {
		b.iter(|| {
			let source = &[I32Const(0), I32Add];

			superoptimize_impl(
				black_box(source),
				black_box(0),
				black_box(&[]),
				black_box(&[ValueType::I32]),
			)
		})
	});
}

fn consts_add(c: &mut Criterion) {
	c.bench_function("consts_add", |b| {
		b.iter(|| {
			let source = &[I32Const(1), I32Const(2), I32Add];

			superoptimize_impl(
				black_box(source),
				black_box(0),
				black_box(&[]),
				black_box(&[]),
			)
		})
	});
}

fn const_nop(c: &mut Criterion) {
	c.bench_function("const_nop", |b| {
		b.iter(|| {
			let source = &[I32Const(1), Nop];

			superoptimize_impl(
				black_box(source),
				black_box(0),
				black_box(&[]),
				black_box(&[]),
			)
		})
	});
}

criterion_group! {
	name = benches;
	config = Criterion::default().sample_size(10);
	targets = const0_add, consts_add, const_nop
}
criterion_main!(benches);
