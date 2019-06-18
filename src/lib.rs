use crate::block::flat_blocks_mut;
use parity_wasm::elements::{
	CodeSection, FunctionSection, Instruction, Type, TypeSection, ValueType,
};
use rayon::prelude::*;
use wassup_encoding::*;
use z3::*;

mod block;

pub fn superoptimize_instructions(
	source_program: &[Instruction],
	n_params: usize,
) -> Vec<Instruction> {
	superoptimize_impl(source_program, n_params)
}

pub fn superoptimize_func_body(
	func_body: &mut parity_wasm::elements::FuncBody,
	params: &[ValueType],
) {
	let n_params = params.len();

	let code = func_body.code_mut().elements_mut();
	let mut blocks = block::parse_blocks(code);
	let flat_blocks = flat_blocks_mut(&mut blocks);
	for flat_block in flat_blocks {
		let instructions = &flat_block[..];
		*flat_block = superoptimize_instructions(instructions, n_params);
	}
	*code = block::serialize(&blocks);
}

pub fn superoptimize_module(module: &mut parity_wasm::elements::Module) {
	// Destructure the module
	// Safe, because the sections should be independent
	let (type_section, code_section, function_section): (
		&TypeSection,
		&mut CodeSection,
		&FunctionSection,
	) = unsafe {
		(
			&*(module.type_section().unwrap() as *const _),
			&mut *(module.code_section_mut().unwrap() as *mut _),
			&*(module.function_section().unwrap() as *const _),
		)
	};

	code_section
		.bodies_mut()
		.par_iter_mut()
		.enumerate()
		.for_each(|(index, body)| {
			let func = function_section.entries()[index];
			let signature = &type_section.types()[func.type_ref() as usize];
			let params = match signature {
				Type::Function(f) => f.params(),
			};
			superoptimize_func_body(body, params)
		});
}

fn superoptimize_impl(source_program: &[Instruction], n_params: usize) -> Vec<Instruction> {
	let config = Config::default();
	let ctx = Context::new(&config);
	let solver = Solver::new(&ctx);

	let constants = Constants::new(
		&ctx,
		&solver,
		stack_depth(source_program) as usize,
		n_params,
	);
	let source_state = State::new(&ctx, &solver, &constants, "source_");
	let target_state = State::new(&ctx, &solver, &constants, "target_");
	source_state.set_source_program(source_program);

	source_state.assert_transitions();
	target_state.assert_transitions();

	// start equivalent
	solver.assert(&equivalent(
		&source_state,
		&ctx.from_u64(0),
		&target_state,
		&ctx.from_u64(0),
	));

	let target_length = &target_state.program_length();

	let mut current_best = source_program.to_vec();

	loop {
		solver.push();

		// force target program to be shorter than current best
		solver.assert(&in_range(
			&ctx.from_u64(0),
			target_length,
			&ctx.from_u64(current_best.len() as u64),
		));
		// assert programs are equivalent
		solver.assert(&equivalent(
			&source_state,
			&ctx.from_u64(source_program.len() as u64),
			&target_state,
			&target_length,
		));

		if !solver.check() {
			// already optimal
			return current_best;
		}

		// better version found
		// decode

		let model = solver.get_model();

		let target_length = model.eval(target_length).unwrap().as_i64().unwrap();
		let mut target_program = Vec::with_capacity(target_length as usize);

		for i in 0..target_length {
			let encoded_instr = model.eval(&target_state.program(&ctx.from_i64(i))).unwrap();

			for instr in iter_intructions() {
				let equal_tester = &constants.instruction_testers[instruction_to_index(instr)];
				let equal = model
					.eval(&equal_tester.apply(&[&encoded_instr]))
					.unwrap()
					.as_bool()
					.unwrap();

				if equal {
					// TODO decode *Local
					let decoded = if let Instruction::I32Const(_) = instr {
						let push_constant_ast = target_state.push_constants(&ctx.from_i64(i));
						let push_constant =
							model.eval(&push_constant_ast).unwrap().as_i64().unwrap();
						// TODO fix cast
						Instruction::I32Const(push_constant as i32)
					} else {
						instr.clone()
					};

					target_program.push(decoded);

					break;
				}
			}
		}

		current_best = target_program;

		solver.pop(1);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn superoptimize_nop() {
		let source_program = &[Instruction::I32Const(1), Instruction::Nop];
		let target = superoptimize_impl(source_program, 0);
		assert_eq!(target, vec![Instruction::I32Const(1)]);
	}

	#[test]
	fn superoptimize_consts_add() {
		let source_program = &[
			Instruction::I32Const(1),
			Instruction::I32Const(2),
			Instruction::I32Add,
		];
		let target = superoptimize_impl(source_program, 0);
		assert_eq!(target, vec![Instruction::I32Const(3)]);
	}

	#[test]
	fn superoptimize_add() {
		let source_program = &[Instruction::I32Const(0), Instruction::I32Add];
		let target = superoptimize_impl(source_program, 0);
		assert_eq!(target, vec![]);
	}

	#[test]
	fn superoptimize_setlocal_0() {
		let source_program = &[Instruction::I32Const(0), Instruction::SetLocal(0)];
		let target = superoptimize_impl(source_program, 0);
		assert_eq!(target, vec![]);
	}

	#[test]
	#[ignore] // TODO
	fn no_superoptimize_setlocal() {
		let source_program = &[Instruction::I32Const(3), Instruction::SetLocal(0)];
		let target = superoptimize_impl(source_program, 1);
		assert_eq!(target, source_program);
	}
}
