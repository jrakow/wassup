mod constants;
mod instructions;
mod state;

pub use crate::{
	constants::Constants,
	instructions::{instruction_to_index, iter_instructions},
	state::State,
};

use crate::instructions::stack_pop_push_count;
use parity_wasm::elements::Instruction;
use z3::*;

pub fn stack_depth(program: &[Instruction]) -> usize {
	let mut stack_pointer: isize = 0;
	let mut lowest: isize = 0;
	for i in program {
		let (pops, pushs) = stack_pop_push_count(i);
		let (pops, pushs) = (pops as isize, pushs as isize);
		lowest = std::cmp::min(lowest, stack_pointer - pops);
		stack_pointer = stack_pointer - pops + pushs;
	}
	lowest.abs() as usize
}

pub fn in_range<'ctx>(a: &Ast<'ctx>, b: &Ast<'ctx>, c: &Ast<'ctx>) -> Ast<'ctx> {
	a.le(&b).and(&[&b.lt(&c)])
}

// this cannot be tested directly as Z3 does not support evaluating terms with quantifiers
pub fn equivalent<'ctx>(
	lhs: &State<'ctx, '_, '_>,
	lhs_pc: &Ast<'ctx>,
	rhs: &State<'ctx, '_, '_>,
	rhs_pc: &Ast<'ctx>,
) -> Ast<'ctx> {
	let ctx = lhs.ctx;

	let stack_pointers_equal = lhs.stack_pointer(&lhs_pc)._eq(&rhs.stack_pointer(&rhs_pc));

	let stacks_equal = {
		// for 0 <= n < stack_pointer
		let n = ctx.named_int_const("n");
		let n_in_range = in_range(&ctx.from_u64(0), &n, &lhs.stack_pointer(&lhs_pc));

		// lhs-stack(lhs_pc, n) ==  rhs-stack(rhs_pc, n)
		let condition = lhs.stack(&lhs_pc, &n)._eq(&rhs.stack(&rhs_pc, &n));

		ctx.forall_const(&[&n], &n_in_range.implies(&condition))
	};

	ctx.from_bool(true)
		.and(&[&stack_pointers_equal, &stacks_equal])
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn stack_depth_test() {
		let program = &[];
		assert_eq!(stack_depth(program), 0);

		let program = &[Instruction::I32Add];
		assert_eq!(stack_depth(program), 2);

		let program = &[Instruction::I32Const(1), Instruction::I32Add];
		assert_eq!(stack_depth(program), 1);

		let program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(1),
			Instruction::I32Const(1),
			Instruction::I32Add,
		];
		assert_eq!(stack_depth(program), 0);
	}
}
