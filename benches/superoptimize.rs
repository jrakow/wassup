use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wassup::superoptimize_snippet;
use wassup_encoding::{Instruction::*, Value::*, ValueTypeConfig};

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

criterion_group! {
	name = benches;
	config = Criterion::default().measurement_time(std::time::Duration::from_secs(60));
	targets = consts_add,
}
criterion_main!(benches);
