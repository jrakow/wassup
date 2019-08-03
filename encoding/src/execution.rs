use crate::{instructions::instruction_datatype, *};
use either::Either;
use parity_wasm::elements::ValueType;
use z3::*;

pub struct Execution<'ctx, 'constants> {
	// for convenience
	ctx: &'ctx Context,
	constants: &'constants Constants<'ctx>,
	program: Vec<Ast<'ctx>>,
	pub states: Vec<EncodedState<'ctx, 'constants>>,
}

impl<'ctx, 'constants, 'solver> Execution<'ctx, 'constants> {
	pub fn new(
		constants: &'constants Constants<'ctx>,
		solver: &Solver<'ctx>,
		prefix: String,
		program: Either<&[Instruction], usize>,
	) -> Self {
		let ctx = constants.ctx;
		let program_length = program.map_left(<[Instruction]>::len).into_inner();

		let states: Vec<_> = (0..program_length + 1)
			.map(|i| EncodedState::new(ctx, constants, prefix.clone(), format!("_{}", i)))
			.collect();

		let instruction_sort = instruction_datatype(ctx, constants.value_type_config).sort;
		let this = Self {
			ctx: constants.ctx,
			constants,
			program: match program {
				Either::Left(program) => program
					.iter()
					.map(|instr| instr.encode(ctx, constants.value_type_config))
					.collect(),
				Either::Right(program_length) => (0..program_length)
					.map(|i| {
						ctx.named_const(&format!("{}program_{}", prefix, i), &instruction_sort)
					})
					.collect(),
			},
			states,
		};

		// initialize
		let initial_state = &this.states[0];

		// set trapped = false
		solver.assert(&initial_state.trapped().not());

		// set stack(i) == initial_stack[i]
		for (i, var) in constants.initial_stack.iter().enumerate() {
			solver.assert(&initial_state.stack(&ctx.from_usize(i))._eq(&var));
		}

		// set stack_counter = initial_stack.len()
		solver.assert(
			&initial_state
				.stack_pointer()
				._eq(&ctx.from_usize(constants.initial_stack.len())),
		);

		// set local(i) = inital_locals[i]
		for (i, var) in constants.initial_locals.iter().enumerate() {
			solver.assert(&initial_state.local(&ctx.from_usize(i))._eq(&var));
		}

		// assert transitions
		let bounds: Vec<_> = constants.bounds.iter().collect();
		for pc in 0..program_length {
			solver.assert(&constants.ctx.forall_const(
				&bounds,
				&transition(
					ctx,
					constants,
					&this.states[pc],
					&this.states[pc + 1],
					&this.program[pc],
				),
			))
		}

		this
	}

	pub fn decode_program(&self, model: &Model) -> Vec<Instruction> {
		self.program
			.iter()
			.map(|ast| Instruction::decode(ast, self.ctx, model, self.constants.value_type_config))
			.collect()
	}
}

fn transition<'ctx, 'constants>(
	ctx: &'ctx Context,
	constants: &'constants Constants<'ctx>,
	state: &EncodedState<'ctx, 'constants>,
	next_state: &EncodedState<'ctx, 'constants>,
	instr: &Ast<'ctx>,
) -> Ast<'ctx> {
	transition_stack(ctx, constants, state, next_state, instr).and(&[&transition_stack_pointer(
		constants, state, next_state, instr,
	)])
}

fn transition_stack_pointer<'ctx, 'constants>(
	constants: &Constants<'ctx>,
	state: &EncodedState<'ctx, 'constants>,
	next_state: &EncodedState<'ctx, 'constants>,
	instr: &Ast<'ctx>,
) -> Ast<'ctx> {
	// encode stack_pointer change
	let stack_pointer = state.stack_pointer();
	let stack_pointer_next = next_state.stack_pointer();

	let pop_count = constants.stack_pop_count(instr);
	let push_count = constants.stack_push_count(instr);

	let new_pointer = stack_pointer.add(&[&push_count]).sub(&[&pop_count]);

	stack_pointer_next._eq(&new_pointer)
}

fn transition_stack<'ctx, 'constants>(
	ctx: &'ctx Context,
	constants: &Constants<'ctx>,
	state: &EncodedState<'ctx, 'constants>,
	next_state: &EncodedState<'ctx, 'constants>,
	instr: &Ast<'ctx>,
) -> Ast<'ctx> {
	use Instruction::*;

	let instruction_datatype = instruction_datatype(ctx, constants.value_type_config);

	let to_i32 = |i: &Ast<'ctx>| constants.value_type_config.i32_wrap_as_i64(ctx, i);
	let as_i32 = |i: &Ast<'ctx>| constants.value_type_config.i64_unwrap_as_i32(i);

	// helpers
	let i32_size = constants.value_type_config.i32_size;
	let i64_size = constants.value_type_config.i64_size;
	let bool_to_i32 = |b: &Ast<'ctx>| {
		b.ite(
			&to_i32(&ctx.from_usize(1).int2bv(i32_size as u64)),
			&to_i32(&ctx.from_usize(0).int2bv(i32_size as u64)),
		)
	};
	let bvmod32 = |b: &Ast<'ctx>| b.bvurem(&ctx.from_usize(i32_size).int2bv(i32_size as u64));
	let bvmod64 = |b: &Ast<'ctx>| {
		let i64_size = constants.value_type_config.i64_size.unwrap();
		b.bvurem(&ctx.from_usize(i64_size).int2bv(i64_size as u64))
	};
	let bv64_zero = || {
		let i64_size = constants.value_type_config.i64_size.unwrap();
		ctx.from_usize(0).int2bv(i64_size as u64)
	};

	let bv32_zero = ctx.from_usize(0).int2bv(i32_size as u64);

	let op = |i| state.stack(&state.stack_pointer().sub(&[&ctx.from_i64(i)]));
	let op_type = |i| state.stack_type(&state.stack_pointer().sub(&[&ctx.from_i64(i)]));
	let result = next_state.stack(&next_state.stack_pointer().sub(&[&ctx.from_i64(1)]));
	let result_type = next_state.stack_type(&next_state.stack_pointer().sub(&[&ctx.from_i64(1)]));
	let i32_type = constants
		.value_type_config
		.encode_value_type(ctx, ValueType::I32);
	let i64_type = || {
		constants
			.value_type_config
			.encode_value_type(ctx, ValueType::I64)
	};

	let mut transitions = Vec::new();
	for (i, variant) in instruction_datatype.variants.iter().enumerate() {
		let active = variant.tester.apply(&[&instr]);
		let template = Instruction::iter_templates(constants.value_type_config)
			.nth(i)
			.unwrap();

		let operand_types_correct = match template {
			I32Eqz => op_type(1)._eq(&i32_type),
			// irelop
			I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU |
			// ibinop
			I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or | I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr
			=> op_type(1)._eq(&i32_type).and(&[&op_type(2)._eq(&i32_type)]),

			I64Eqz => op_type(1)._eq(&i64_type()),
			// irelop
			I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS | I64GeU |
			// ibinop
			I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And | I64Or | I64Xor | I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr
			=> op_type(1)._eq(&i64_type()).and(&[&op_type(2)._eq(&i64_type())]),

			Select => op_type(1)._eq(&i32_type).and(&[&op_type(2)._eq(&op_type(3))]),
			SetLocal(_) | TeeLocal(_) => {
				let index = variant.accessors[0].apply(&[&instr]);
				let ty = constants.local_type(&index);
				op_type(1)._eq(&ty)
			},
			// conversions
			I32WrapI64 => op_type(1)._eq(&i64_type()),
			I64ExtendSI32 | I64ExtendUI32 => op_type(1)._eq(&i32_type),

			_ => ctx.from_bool(true),
		};

		let result_type_correct = match template {
			I32Eqz | I64Eqz |
			// irelop 32
			I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU |
			// irelop 64
			I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS | I64GeU |
			// ibinop 32
			I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or | I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr
			=> result_type._eq(&i32_type),

			// ibinop
			I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And | I64Or | I64Xor | I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr
			=> result_type._eq(&i64_type()),

			Select => result_type._eq(&op_type(2)),
			Const(_) => result_type._eq(&variant.accessors[1].apply(&[&instr])),
			GetLocal(_) | TeeLocal(_) => {
				let index = variant.accessors[0].apply(&[&instr]);
				let ty = constants.local_type(&index);
				result_type._eq(&ty)
			}
			// conversions
			I32WrapI64 => result_type._eq(&i32_type),
			I64ExtendSI32 | I64ExtendUI32 => result_type._eq(&i64_type()),

			_ => ctx.from_bool(true),
		};

		let instr_traps = match template {
			Unreachable => ctx.from_bool(true),

			I32DivU | I32RemU | I32RemS => as_i32(&op(2))._eq(&bv32_zero),
			I32DivS => {
				let divide_by_zero = as_i32(&op(2))._eq(&bv32_zero);
				let overflow = as_i32(&op(2)).bvsdiv_no_overflow(&as_i32(&op(1)));
				divide_by_zero.or(&[&overflow])
			}
			I64DivU | I64RemU | I64RemS => op(2)._eq(&bv64_zero()),
			I64DivS => {
				let divide_by_zero = op(2)._eq(&bv64_zero());
				let overflow = &op(2).bvsdiv_no_overflow(&op(1));
				divide_by_zero.or(&[&overflow])
			}
			_ => ctx.from_bool(false),
		};
		let next_trapped = state.trapped().or(&[&instr_traps]);
		let trapped_transition = next_state.trapped()._eq(&next_trapped);

		// TODO simplify
		let locals_preserved = if !constants.initial_locals.is_empty() {
			// preserve all locals which are not set in this step
			let i = &ctx.named_int_const("i");
			let i_in_range = in_range(&ctx.from_usize(0), &i, &constants.n_locals);

			let index_active = match template {
				SetLocal(_) | TeeLocal(_) => {
					let index = variant.accessors[0].apply(&[&instr]);
					index._eq(&i)
				}
				_ => ctx.from_bool(false),
			};

			ctx.forall_const(
				&[&i],
				&i_in_range
					.and(&[&index_active.not()])
					.implies(&next_state.local(&i)._eq(&state.local(&i))),
			)
		} else {
			ctx.from_bool(true)
		};

		// TODO test
		let no_stack_underflow = {
			// stack_pointer >= pops
			let pops = template.stack_pop_push_count().0;
			state.stack_pointer().ge(&ctx.from_usize(pops))
		};

		let stack_preserved = {
			// preserve stack addresses 0 ..= stack_pointer - pops - 1
			let n = ctx.named_int_const("n");

			let n_in_range = in_range(
				&ctx.from_usize(0),
				&n,
				&state
					.stack_pointer()
					.sub(&[&constants.stack_pop_count(&instr)]),
			);
			let slot_preserved = state.stack(&n)._eq(&next_state.stack(&n));
			let type_preserved = state.stack_type(&n)._eq(&next_state.stack_type(&n));

			// forall n
			ctx.forall_const(
				&[&n],
				&n_in_range.implies(&slot_preserved.and(&[&type_preserved])),
			)
		};

		let local_index_in_range = match template {
			GetLocal(_) | SetLocal(_) | TeeLocal(_) => {
				let index = variant.accessors[0].apply(&[&instr]);
				in_range(&ctx.from_usize(0), &index, &constants.n_locals)
			}
			_ => ctx.from_bool(true),
		};

		let transition = match template {
			Unreachable | Nop | Drop => ctx.from_bool(true),

			Const(_) => result._eq(&variant.accessors[0].apply(&[&instr])),

			I32Eqz => result._eq(&bool_to_i32(&as_i32(&op(1))._eq(&bv32_zero))),
			I32Eq => result._eq(&bool_to_i32(&as_i32(&op(2))._eq(&as_i32(&op(1))))),
			I32Ne => result._eq(&bool_to_i32(&as_i32(&op(2))._eq(&as_i32(&op(1))).not())),
			I32LtS => result._eq(&bool_to_i32(&as_i32(&op(2)).bvslt(&as_i32(&op(1))))),
			I32LtU => result._eq(&bool_to_i32(&as_i32(&op(2)).bvult(&as_i32(&op(1))))),
			I32GtS => result._eq(&bool_to_i32(&as_i32(&op(2)).bvsgt(&as_i32(&op(1))))),
			I32GtU => result._eq(&bool_to_i32(&as_i32(&op(2)).bvugt(&as_i32(&op(1))))),
			I32LeS => result._eq(&bool_to_i32(&as_i32(&op(2)).bvsle(&as_i32(&op(1))))),
			I32LeU => result._eq(&bool_to_i32(&as_i32(&op(2)).bvule(&as_i32(&op(1))))),
			I32GeS => result._eq(&bool_to_i32(&as_i32(&op(2)).bvsge(&as_i32(&op(1))))),
			I32GeU => result._eq(&bool_to_i32(&as_i32(&op(2)).bvuge(&as_i32(&op(1))))),

			I64Eqz => result._eq(&bool_to_i32(&op(1)._eq(&bv64_zero()))),
			I64Eq => result._eq(&bool_to_i32(&op(2)._eq(&op(1)))),
			I64Ne => result._eq(&bool_to_i32(&op(2)._eq(&op(1)).not())),
			I64LtS => result._eq(&bool_to_i32(&op(2).bvslt(&op(1)))),
			I64LtU => result._eq(&bool_to_i32(&op(2).bvult(&op(1)))),
			I64GtS => result._eq(&bool_to_i32(&op(2).bvsgt(&op(1)))),
			I64GtU => result._eq(&bool_to_i32(&op(2).bvugt(&op(1)))),
			I64LeS => result._eq(&bool_to_i32(&op(2).bvsle(&op(1)))),
			I64LeU => result._eq(&bool_to_i32(&op(2).bvule(&op(1)))),
			I64GeS => result._eq(&bool_to_i32(&op(2).bvsge(&op(1)))),
			I64GeU => result._eq(&bool_to_i32(&op(2).bvuge(&op(1)))),

			I32Add => result._eq(&to_i32(&as_i32(&op(2)).bvadd(&as_i32(&op(1))))),
			I32Sub => result._eq(&to_i32(&as_i32(&op(2)).bvsub(&as_i32(&op(1))))),
			I32Mul => result._eq(&to_i32(&as_i32(&op(2)).bvmul(&as_i32(&op(1))))),
			I32DivS => result._eq(&to_i32(&as_i32(&op(2)).bvsdiv(&as_i32(&op(1))))),
			I32DivU => result._eq(&to_i32(&as_i32(&op(2)).bvudiv(&as_i32(&op(1))))),
			I32RemS => result._eq(&to_i32(&as_i32(&op(2)).bvsrem(&as_i32(&op(1))))),
			I32RemU => result._eq(&to_i32(&as_i32(&op(2)).bvurem(&as_i32(&op(1))))),
			I32And => result._eq(&to_i32(&as_i32(&op(2)).bvand(&as_i32(&op(1))))),
			I32Or => result._eq(&to_i32(&as_i32(&op(2)).bvor(&as_i32(&op(1))))),
			I32Xor => result._eq(&to_i32(&as_i32(&op(2)).bvxor(&as_i32(&op(1))))),
			I32Shl => result._eq(&to_i32(&as_i32(&op(2)).bvshl(&bvmod32(&as_i32(&op(1)))))),
			I32ShrS => result._eq(&to_i32(&as_i32(&op(2)).bvashr(&bvmod32(&as_i32(&op(1)))))),
			I32ShrU => result._eq(&to_i32(&as_i32(&op(2)).bvlshr(&bvmod32(&as_i32(&op(1)))))),
			I32Rotl => result._eq(&to_i32(&as_i32(&op(2)).bvrotl(&bvmod32(&as_i32(&op(1)))))),
			I32Rotr => result._eq(&to_i32(&as_i32(&op(2)).bvrotr(&bvmod32(&as_i32(&op(1)))))),

			I64Add => result._eq(&op(2).bvadd(&op(1))),
			I64Sub => result._eq(&op(2).bvsub(&op(1))),
			I64Mul => result._eq(&op(2).bvmul(&op(1))),
			I64DivS => result._eq(&op(2).bvsdiv(&op(1))),
			I64DivU => result._eq(&op(2).bvudiv(&op(1))),
			I64RemS => result._eq(&op(2).bvsrem(&op(1))),
			I64RemU => result._eq(&op(2).bvurem(&op(1))),
			I64And => result._eq(&op(2).bvand(&op(1))),
			I64Or => result._eq(&op(2).bvor(&op(1))),
			I64Xor => result._eq(&op(2).bvxor(&op(1))),
			I64Shl => result._eq(&op(2).bvshl(&bvmod64(&op(1)))),
			I64ShrS => result._eq(&op(2).bvashr(&bvmod64(&op(1)))),
			I64ShrU => result._eq(&op(2).bvlshr(&bvmod64(&op(1)))),
			I64Rotl => result._eq(&op(2).bvrotl(&bvmod64(&op(1)))),
			I64Rotr => result._eq(&op(2).bvrotr(&bvmod64(&op(1)))),

			// conversions
			I32WrapI64 => result._eq(&to_i32(&op(1).bvextract(i32_size - 1, 0))),
			I64ExtendSI32 => result._eq(&as_i32(&op(1)).bvsignextend(i64_size.unwrap() - i32_size)),
			I64ExtendUI32 => result._eq(&as_i32(&op(1)).bvzeroextend(i64_size.unwrap() - i32_size)),

			Select => result._eq(&as_i32(&op(1))._eq(&bv32_zero).ite(&op(2), &op(3))),

			GetLocal(_) => {
				let index = variant.accessors[0].apply(&[&instr]);

				result._eq(&state.local(&index))
			}
			SetLocal(_) => {
				let index = variant.accessors[0].apply(&[&instr]);

				next_state.local(&index)._eq(&op(1))
			}
			TeeLocal(_) => {
				let index = variant.accessors[0].apply(&[&instr]);

				let local_is_set = next_state.local(&index)._eq(&op(1));
				let stack_is_set = result._eq(&op(1));

				local_is_set.and(&[&stack_is_set])
			}
		};

		transitions.push(active.implies(&transition.and(&[
			&operand_types_correct,
			&result_type_correct,
			&trapped_transition,
			&locals_preserved,
			&stack_preserved,
			&no_stack_underflow,
			&local_index_in_range,
		])));
	}

	// create vector of references
	let transitions: Vec<_> = transitions.iter().collect();
	ctx.from_bool(true).and(&transitions)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{Value::*, ValueTypeConfig};
	use Instruction::*;

	#[test]
	fn program_encode_decode() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Nop, Const(I32(2)), I32Add];

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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(&execution.decode_program(&model), program);
	}

	#[test]
	fn initial_conditions() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Const(I32(2)), I32Add];

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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			execution.states[0].decode(&model, &constants),
			State {
				stack: vec![],
				locals: vec![],
				trapped: false,
			}
		);
	}

	#[test]
	fn transition_const() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1))];

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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			execution.states[1].decode(&model, &constants),
			State {
				stack: vec![I32(1)],
				locals: vec![],
				trapped: false,
			}
		);
	}

	#[test]
	fn transition_add_consts() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Nop, Const(I32(2)), I32Add];

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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			execution.states[1].decode(&model, &constants),
			State {
				stack: vec![I32(1)],
				locals: vec![],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[2].decode(&model, &constants),
			State {
				stack: vec![I32(1)],
				locals: vec![],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[3].decode(&model, &constants),
			State {
				stack: vec![I32(1), I32(2)],
				locals: vec![],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[4].decode(&model, &constants),
			State {
				stack: vec![I32(3)],
				locals: vec![],
				trapped: false,
			}
		);
	}

	#[test]
	fn transition_consts_eq() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I64(15)), Const(I64(15)), I64Eq];

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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			execution.states[3].decode(&model, &constants),
			State {
				stack: vec![I32(1)],
				locals: vec![],
				trapped: false,
			}
		);
	}

	#[test]
	fn transition_consts_eq_false() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I64(1)), Const(I64(2)), I64Eq];

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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			execution.states[3].decode(&model, &constants),
			State {
				stack: vec![I32(0)],
				locals: vec![],
				trapped: false,
			}
		);
	}

	#[test]
	fn transition_add_consts_config_i32() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Nop, Const(I32(2)), I32Add];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: None,
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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			execution.states[1].decode(&model, &constants),
			State {
				stack: vec![I32(1)],
				locals: vec![],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[2].decode(&model, &constants),
			State {
				stack: vec![I32(1)],
				locals: vec![],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[3].decode(&model, &constants),
			State {
				stack: vec![I32(1), I32(2)],
				locals: vec![],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[4].decode(&model, &constants),
			State {
				stack: vec![I32(3)],
				locals: vec![],
				trapped: false,
			}
		);
	}

	#[test]
	fn transition_add_consts_config_reduced_size() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I64(1)), Nop, Const(I64(2)), I64Add];

		let value_type_config = ValueTypeConfig {
			i32_size: 8,
			i64_size: Some(16),
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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			execution.states[1].decode(&model, &constants),
			State {
				stack: vec![I64(1)],
				locals: vec![],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[2].decode(&model, &constants),
			State {
				stack: vec![I64(1)],
				locals: vec![],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[3].decode(&model, &constants),
			State {
				stack: vec![I64(1), I64(2)],
				locals: vec![],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[4].decode(&model, &constants),
			State {
				stack: vec![I64(3)],
				locals: vec![],
				trapped: false,
			}
		);
	}

	#[test]
	fn transition_const_drop() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Drop];

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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			execution.states[2].decode(&model, &constants),
			State {
				stack: vec![],
				locals: vec![],
				trapped: false,
			}
		);
	}

	#[test]
	fn transition_consts_select_true() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Const(I32(2)), Const(I32(3)), Select];

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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			execution.states[4].decode(&model, &constants),
			State {
				stack: vec![I32(1)],
				locals: vec![],
				trapped: false,
			}
		);
	}

	#[test]
	fn transition_consts_select_false() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(1)), Const(I32(2)), Const(I32(0)), Select];

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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			execution.states[4].decode(&model, &constants),
			State {
				stack: vec![I32(2)],
				locals: vec![],
				trapped: false,
			}
		);
	}

	#[test]
	fn transition_local() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[
			// x2 = x1 + x0
			GetLocal(0),
			GetLocal(1),
			I32Add,
			SetLocal(2),
			// swap x0, x1
			// tmp = x1; x1 = x0; x0 = tmp;
			GetLocal(1),
			GetLocal(0),
			SetLocal(1),
			TeeLocal(0),
		];

		let value_type_config = ValueTypeConfig {
			i32_size: 32,
			i64_size: Some(64),
		};

		let initial_locals = vec![
			I32(1).encode(&ctx, value_type_config),
			I32(2).encode(&ctx, value_type_config),
			I32(0).encode(&ctx, value_type_config),
		];

		let constants = Constants::new(
			&ctx,
			&solver,
			vec![],
			initial_locals,
			vec![ValueType::I32; 3],
			&[],
			value_type_config,
		);
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert_eq!(
			execution.states[0].decode(&model, &constants),
			State {
				stack: vec![],
				locals: vec![I32(1), I32(2), I32(0)],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[1].decode(&model, &constants),
			State {
				stack: vec![I32(1)],
				locals: vec![I32(1), I32(2), I32(0)],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[2].decode(&model, &constants),
			State {
				stack: vec![I32(1), I32(2)],
				locals: vec![I32(1), I32(2), I32(0)],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[3].decode(&model, &constants),
			State {
				stack: vec![I32(3)],
				locals: vec![I32(1), I32(2), I32(0)],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[4].decode(&model, &constants),
			State {
				stack: vec![],
				locals: vec![I32(1), I32(2), I32(3)],
				trapped: false,
			}
		);
		assert_eq!(
			execution.states[8].decode(&model, &constants),
			State {
				stack: vec![I32(2)],
				locals: vec![I32(2), I32(1), I32(3)],
				trapped: false,
			}
		);
	}

	#[test]
	fn transition_trapped_unreachable() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Unreachable, Nop];

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
		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert!(!execution.states[0].decode(&model, &constants).trapped);
		assert!(execution.states[1].decode(&model, &constants).trapped);
		assert!(execution.states[2].decode(&model, &constants).trapped);
	}

	#[test]
	fn transition_trapped_div0() {
		let ctx = Context::new(&Config::default());
		let solver = Solver::new(&ctx);

		let program = &[Const(I32(0)), Const(I32(1)), I32DivU];

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

		let execution = Execution::new(&constants, &solver, "".to_owned(), Either::Left(program));

		assert!(solver.check());
		let model = solver.get_model();

		assert!(!execution.states[0].decode(&model, &constants).trapped);
		assert!(!execution.states[1].decode(&model, &constants).trapped);
		assert!(!execution.states[2].decode(&model, &constants).trapped);
		assert!(execution.states[3].decode(&model, &constants).trapped);
	}
}
