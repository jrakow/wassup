use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wassup_encoding::*;
use z3::*;

fn transition_consts_add(source: &[Instruction]) {
	let ctx = Context::new(&Config::default());
	let solver = Solver::new(&ctx);

	let value_type_config = ValueTypeConfig {
		i32_size: 32,
		i64_size: Some(64),
	};
	let constants = Constants::new(&ctx, &solver, vec![], vec![], &[], value_type_config);
	let state = State::new(&ctx, &solver, &constants, "");

	state.set_source_program(source);

	solver.assert(&state.transitions());

	assert!(solver.check());
	let model = solver.get_model();

	let stack = |pc: usize, i: usize| -> Value {
		let value = &state.stack(&ctx.from_usize(pc), &ctx.from_usize(i));
		let ty = &state.stack_type(&ctx.from_usize(pc), &ctx.from_usize(i));

		Value::decode(
			value,
			&model,
			value_type_config.decode_value_type(&ctx, &model, ty),
			value_type_config,
		)
	};

	assert_eq!(stack(1, 0), Value::I32(1));
	assert_eq!(stack(2, 0), Value::I32(1));
	assert_eq!(stack(2, 1), Value::I32(2));
	assert_eq!(stack(3, 0), Value::I32(3));
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
