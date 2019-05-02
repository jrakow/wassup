use parity_wasm::elements::Instruction;
use std::convert::TryInto;
use wassup_z3::*;

struct Constants<'ctx> {
	word_sort: Sort<'ctx>,
	int_sort: Sort<'ctx>,
	instruction_sort: Sort<'ctx>,
	instruction_consts: Vec<FuncDecl<'ctx>>,
	stack_pop_count_func: FuncDecl<'ctx>,
	stack_push_count_func: FuncDecl<'ctx>,
	initial_stack: Vec<Ast<'ctx>>,
	stack_depth: usize,
}

impl<'ctx> Constants<'ctx> {
	fn instruction(&'ctx self, i: &Instruction) -> Ast<'ctx> {
		use Instruction::*;

		match i {
			I32Const(_) => self.instruction_consts[0].apply(&[]),
			I32Add => self.instruction_consts[1].apply(&[]),
			_ => unimplemented!(),
		}
	}
}

fn create_constants<'ctx>(
	ctx: &'ctx Context,
	solver: &Solver,
	stack_depth: usize,
) -> Constants<'ctx> {
	use Instruction::*;

	let word_sort = ctx.bv_sort(32);
	let int_sort = ctx.int_sort();
	let (instruction_sort, instruction_consts, _) = ctx.enumeration_sort(
		&ctx.string_symbol("instruction-sort"),
		&[&ctx.string_symbol("I32Const"), &ctx.string_symbol("I32Add")],
	);
	let initial_stack: Vec<_> = (0..stack_depth)
		.map(|i| ctx.fresh_const("initial-stack", word_sort))
		.collect();

	let stack_pop_count_func = ctx.func_decl(
		&ctx.string_symbol("stack-pop-count"),
		&[instruction_sort],
		int_sort,
	);
	let stack_push_count_func = ctx.func_decl(
		&ctx.string_symbol("stack-push-count"),
		&[instruction_sort],
		int_sort,
	);

	let constants = Constants {
		word_sort,
		int_sort,
		instruction_sort,
		instruction_consts,
		stack_pop_count_func,
		stack_push_count_func,
		initial_stack,
		stack_depth,
	};

	let int = |i| ctx.int(i, int_sort);
	let pop_count = |i| {
		constants
			.stack_pop_count_func
			.apply(&[constants.instruction(i)])
	};
	let push_count = |i| {
		constants
			.stack_push_count_func
			.apply(&[constants.instruction(i)])
	};
	solver.assert(&pop_count(&I32Const(0)).eq(&int(0)));
	solver.assert(&push_count(&I32Const(0)).eq(&int(1)));
	solver.assert(&pop_count(&I32Add).eq(&int(2)));
	solver.assert(&push_count(&I32Add).eq(&int(1)));

	constants
}

struct State<'ctx> {
	stack_func: FuncDecl<'ctx>,
	stack_pointer_func: FuncDecl<'ctx>,
	program_func: FuncDecl<'ctx>,
	program_length: usize,
}

fn create_state<'ctx>(
	ctx: &'ctx Context,
	constants: &'ctx Constants,
	prefix: &str,
	program_length: usize,
) -> State<'ctx> {
	let instruction_sort = constants.instruction_sort;

	// declare stack function
	let stack_func = ctx.func_decl(
		&ctx.string_symbol(&format!("{}stack", prefix)),
		&[
			constants.int_sort, // instruction counter
			constants.int_sort, // stack address
		],
		constants.word_sort,
	);

	// declare stack pointer function
	let stack_pointer_func = ctx.func_decl(
		&ctx.string_symbol(&format!("{}stack-pointer", prefix)),
		&[constants.int_sort],
		constants.int_sort,
	);

	// declare program function
	let program_func = ctx.func_decl(
		&ctx.string_symbol(&format!("{}program", prefix)),
		&[constants.int_sort],
		constants.instruction_sort,
	);

	State {
		stack_func,
		stack_pointer_func,
		program_func,
		program_length,
	}
}

fn set_initial_state(ctx: &Context, solver: &Solver, constants: &Constants, state: &State) {
	// set stack(0, i) == xs[i]
	for (i, var) in constants.initial_stack.iter().enumerate() {
		let lhs = state.stack_func.apply(&[
			ctx.int(0, constants.int_sort),
			ctx.int(i, constants.int_sort),
		]);
		solver.assert(&lhs.eq(var));
	}

	// set stack_counter(0) = 0
	solver.assert(
		&state
			.stack_pointer_func
			.apply(&[ctx.int(0, constants.int_sort)])
			.eq(&ctx.int(0, constants.int_sort)),
	);
}

fn set_source_program(
	ctx: &Context,
	solver: &Solver,
	constants: &Constants,
	state: &State,
	program: &[Instruction],
) {
	// set program_func to program
	for (index, instruction) in program.iter().enumerate() {
		let index = ctx.int(index, constants.int_sort);
		let instruction = constants.instruction(instruction);
		solver.assert(&state.program_func.apply(&[index]).eq(&instruction))
	}
}

fn gas_particle_cost(_: &Instruction) -> usize {
	1
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn stack_pop_push_count() {
		let ctx = {
			let cfg = Config::default();
			Context::with_config(&cfg)
		};
		let solver = Solver::with_context(&ctx);
		let constants = create_constants(&ctx, &solver, 0);

		assert!(solver.check());
		let model = solver.model();

		let eval_count = |func: &FuncDecl, i| {
			let ast = func.apply(&[constants.instruction(i)]);
			let ast = model.eval(&ast);
			let i: i64 = (&ast).try_into().unwrap();
			i
		};

		assert_eq!(
			eval_count(&constants.stack_pop_count_func, &Instruction::I32Add),
			2
		);
		assert_eq!(
			eval_count(&constants.stack_push_count_func, &Instruction::I32Add),
			1
		);
		assert_eq!(
			eval_count(&constants.stack_pop_count_func, &Instruction::I32Const(0)),
			0
		);
		assert_eq!(
			eval_count(&constants.stack_push_count_func, &Instruction::I32Const(0)),
			1
		);
	}

	#[test]
	fn simple() {
		let ctx = {
			let cfg = Config::default();
			Context::with_config(&cfg)
		};
		let solver = Solver::with_context(&ctx);

		let program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(2),
			Instruction::I32Add,
		];

		let constants = create_constants(&ctx, &solver, 2);
		let state = create_state(&ctx, &constants, "", program.len());
		set_initial_state(&ctx, &solver, &constants, &state);

		set_source_program(&ctx, &solver, &constants, &state, program);
		println!("{}", &solver);

		assert!(solver.check());
		let model = solver.model();
		println!("{}", model);
	}
}
