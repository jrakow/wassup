use criterion::{black_box, criterion_group, criterion_main, Criterion};
use either::Either;
use wassup_encoding::*;
use z3::*;

fn transition_consts_add(source: &[Instruction]) {
	let ctx = Context::new(&Config::default());
	let solver = Solver::new(&ctx);

	let value_type_config = ValueTypeConfig {
		i32_size: 32,
		i64_size: Some(64),
	};
	let constants = Constants::new(
		&ctx,
		&solver,
		vec![],
		vec![],
		vec![],
		&[],
		value_type_config,
	);
	let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(source));

	assert!(solver.check());
	let model = solver.get_model();

	assert_eq!(
		execution.states[3].decode(&model, &constants),
		State {
			stack: vec![Value::I32(3)],
			locals: vec![],
			trapped: false,
		}
	);
}

fn consts_add(c: &mut Criterion) {
	c.bench_function("consts_add", |b| {
		b.iter(|| {
			let source = &[
				Instruction::Const(Value::I32(1)),
				Instruction::Const(Value::I32(2)),
				Instruction::I32Add,
			];

			transition_consts_add(black_box(source))
		})
	});
}

criterion_group! {
	name = benches;
	config = Criterion::default().sample_size(50);
	targets = consts_add
}
criterion_main!(benches);
