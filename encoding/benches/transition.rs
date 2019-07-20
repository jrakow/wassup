use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wassup_encoding::*;
use z3::*;

fn transition_consts_add(source: &[Instruction]) {
	let ctx = Context::new(&Config::default());
	let solver = Solver::new(&ctx);

	let value_type_config = ValueTypeConfig::Mixed(32, 64);
	let constants = Constants::new(&ctx, vec![], &[], value_type_config);
	let state = State::new(&ctx, &solver, &constants, "");

	state.set_source_program(source);

	solver.assert(&state.transitions());

	assert!(solver.check());
	let model = solver.get_model();

	let value_type = value_type_config.value_type(&ctx);
	let stack = |pc, i| -> i32 {
		let pc = &ctx.from_usize(pc);
		let i = &ctx.from_usize(i);
		let ast = &state.stack(pc, i);

		let inner = value_type.variants[0].accessors[0].apply(&[ast]);
		let evaled = model.eval(&inner.bv2int(true)).unwrap();
		evaled.as_i32().unwrap()
	};

	assert_eq!(stack(1, 0), 1);
	assert_eq!(stack(2, 0), 1);
	assert_eq!(stack(2, 1), 2);
	assert_eq!(stack(3, 0), 3);
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
