use parity_wasm::elements::Instruction;
use std::convert::TryInto;
use std::ffi::{CStr, CString};
use std::iter::once;
use std::ptr::{null, null_mut};
use wassup_z3::*;

fn encode_init_conditions(ctx: &Context, solver: &Solver, program: &[Instruction]) {
	let word_sort = ctx.bv_sort(32);
	let int_sort = ctx.int_sort();
	let sort_name = ctx.string_symbol("instruction_sort");
	let (instruction_sort, instruction_consts, instruction_testers) = ctx.enumeration_sort(
		&ctx.string_symbol("instruction_sort"),
		&[&ctx.string_symbol("I32Const"), &ctx.string_symbol("I32Add")],
	);
	let mk_instruction = |i: &Instruction| {
		let index = match i {
			Instruction::I32Const(_) => 0,
			Instruction::I32Add => 1,
			_ => unimplemented!(),
		};
		instruction_consts[index].apply(&[])
	};

	let stack_depth = 2;
	let stack_vars: Vec<_> = (0..stack_depth)
		.map(|i| ctx.fresh_const("initial-stack", &word_sort))
		.collect();

	// declare stack function
	let stack_func_name = ctx.string_symbol("stack");

	let stack_func_domain: Vec<_> = (0..stack_depth)
		.map(|_| word_sort) // stack variables
		.chain(once(int_sort)) // instruction counter
		.chain(once(int_sort)) // stack address
		.collect();
	let stack_func = ctx.func_decl(&ctx.string_symbol("stack"), &stack_func_domain, &word_sort);

	// set stack(xs, 0, i) == xs[i]
	let program_counter = ctx.int(0, &int_sort);
	for i in 0..stack_depth {
		let stack_index = ctx.int(i, &int_sort);
		let args: Vec<_> = stack_vars
			.iter()
			.cloned()
			.chain(once(program_counter.clone()))
			.chain(once(stack_index))
			.collect();
		let lhs = stack_func.apply(&args);
		let rhs = &stack_vars[i];
		solver.assert(&lhs.eq(rhs));
	}

	// declare stack pointer function
	let stack_pointer_func =
		ctx.func_decl(&ctx.string_symbol("stack_pointer"), &[int_sort], &int_sort);

	// set stack_counter(0) = 0
	solver.assert(
		&stack_pointer_func
			.apply(&[ctx.int(0, &int_sort)])
			.eq(&ctx.int(0, &int_sort)),
	);

	// declare program function
	let program_func = ctx.func_decl(
		&ctx.string_symbol("program"),
		&[int_sort],
		&instruction_sort,
	);
	// set program to program
	for (index, instruction) in program.iter().enumerate() {
		solver.assert(
			&mk_instruction(instruction).eq(&program_func.apply(&[ctx.int(index, &int_sort)])),
		)
	}
	let program_length = program.len();

	// encode transition
	for i in 0..program_length {
		// encode stack_pointer change
		let stack_counter_i = stack_pointer_func.apply(&[ctx.int(i, &int_sort)]);
		let stack_counter_ii = stack_pointer_func.apply(&[ctx.int(i + 1, &int_sort)]);

		let (pop_count, push_count) = stack_pop_push_count(&program[i]);
		let pop_count = ctx.int(pop_count, &int_sort);
		let push_count = ctx.int(push_count, &int_sort);

		let new_counter = &(&stack_counter_i - &pop_count) + &push_count;
		solver.assert(&stack_counter_ii.eq(&new_counter));
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

		encode_init_conditions(
			&ctx,
			&solver,
			&[
				Instruction::I32Const(1),
				Instruction::I32Const(2),
				Instruction::I32Add,
			],
		);
		solver.check();
		let model = solver.model();
		println!("{}", model);
	}
}
