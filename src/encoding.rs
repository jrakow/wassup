use parity_wasm::elements::Instruction;
use wassup_z3::*;

struct Constants<'ctx> {
	word_sort: Sort<'ctx>,
	int_sort: Sort<'ctx>,
	instruction_sort: Sort<'ctx>,
	instruction_consts: Vec<FuncDecl<'ctx>>,
	initial_stack: Vec<Ast<'ctx>>,
	stack_depth: usize,
}

fn create_instruction<'ctx>(constants: &'ctx Constants, i: &Instruction) -> Ast<'ctx> {
	use Instruction::*;

	match i {
		I32Const(_) => constants.instruction_consts[0].apply(&[]),
		I32Add => constants.instruction_consts[1].apply(&[]),
		_ => unimplemented!(),
	}
}

fn create_constants(ctx: &Context, stack_depth: usize) -> Constants {
	let word_sort = ctx.bv_sort(32);
	let int_sort = ctx.int_sort();
	let (instruction_sort, instruction_consts, _) = ctx.enumeration_sort(
		&ctx.string_symbol("instruction-sort"),
		&[&ctx.string_symbol("I32Const"), &ctx.string_symbol("I32Add")],
	);
	let initial_stack: Vec<_> = (0..stack_depth)
		.map(|i| ctx.fresh_const("initial-stack", &word_sort))
		.collect();

	Constants {
		word_sort,
		int_sort,
		instruction_sort,
		instruction_consts,
		initial_stack,
		stack_depth,
	}
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
		&constants.word_sort,
	);

	// declare stack pointer function
	let stack_pointer_func = ctx.func_decl(
		&ctx.string_symbol(&format!("{}stack_pointer", prefix)),
		&[constants.int_sort],
		&constants.int_sort,
	);

	// declare program function
	let program_func = ctx.func_decl(
		&ctx.string_symbol(&format!("{}program", prefix)),
		&[constants.int_sort],
		&constants.instruction_sort,
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
			ctx.int(0, &constants.int_sort),
			ctx.int(i, &constants.int_sort),
		]);
		solver.assert(&lhs.eq(var));
	}

	// set stack_counter(0) = 0
	solver.assert(
		&state
			.stack_pointer_func
			.apply(&[ctx.int(0, &constants.int_sort)])
			.eq(&ctx.int(0, &constants.int_sort)),
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
		let index = ctx.int(index, &constants.int_sort);
		let instruction = create_instruction(constants, instruction);
		solver.assert(&state.program_func.apply(&[index]).eq(&instruction))
	}
}

/// Number of items popped / pushed by the instruction
fn stack_pop_push_count(i: &Instruction) -> (usize, usize) {
	use Instruction::*;

	match i {
		I32Const(_) => (0, 1),
		I32Add => (2, 1),
		_ => unimplemented!(),
	}
}

fn gas_particle_cost(_: &Instruction) -> usize {
	1
}

#[cfg(test)]
mod tests {
	use super::*;

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

		let constants = create_constants(&ctx, 2);
		let state = create_state(&ctx, &constants, "", program.len());
		set_initial_state(&ctx, &solver, &constants, &state);

		set_source_program(&ctx, &solver, &constants, &state, program);

		solver.check();
		let model = solver.model();
		println!("{}", model);
	}
}
