use either::Either;
use parity_wasm::elements::{CodeSection, FunctionSection, Type, TypeSection, ValueType};
use rayon::prelude::*;
use wassup_encoding::*;
use z3::*;

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

pub fn superoptimize_func_body(
	func_body: &mut parity_wasm::elements::FuncBody,
	params: &[ValueType],
) {
	let mut function = Function::from_wasm_func_body_params(func_body, params);
	for snippet in function.instructions.iter_mut() {
		if let Either::Left(vec) = snippet {
			let optimized = superoptimize_snippet(&vec, function.n_params);
			*vec = optimized;
		}
	}

	*func_body = function.to_wasm_func_body();
}

pub fn superoptimize_snippet(source_program: &[Instruction], n_params: usize) -> Vec<Instruction> {
	let config = Config::default();
	let ctx = Context::new(&config);
	let solver = Solver::new(&ctx);

	// TODO compute actual stack types
	let initial_stack = vec![ValueType::I32; stack_depth(&source_program[..])];
	let constants = Constants::new(&ctx, n_params, &initial_stack);
	let source_state = State::new(&ctx, &solver, &constants, "source_");
	let target_state = State::new(&ctx, &solver, &constants, "target_");
	source_state.set_source_program(&source_program[..]);

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
			&ctx.from_usize(current_best.len()),
		));
		// assert programs are equivalent
		solver.assert(&equivalent(
			&source_state,
			&ctx.from_usize(source_program.len()),
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
		current_best = target_state.decode_program(&model);

		solver.pop(1);
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use Instruction::*;

	#[test]
	fn superoptimize_nop() {
		let source_program = &[I32Const(1), Nop];
		let target = superoptimize_snippet(source_program, 0);
		assert_eq!(target, vec![I32Const(1)]);
	}

	#[test]
	fn superoptimize_consts_add() {
		let source_program = &[I32Const(1), I32Const(2), I32Add];
		let target = superoptimize_snippet(source_program, 0);
		assert_eq!(target, vec![I32Const(3)]);
	}

	#[test]
	fn superoptimize_add() {
		let source_program = &[I32Const(0), I32Add];
		let target = superoptimize_snippet(source_program, 0);
		assert_eq!(target, vec![]);
	}

	#[test]
	fn superoptimize_setlocal_0() {
		let source_program = &[I32Const(0), I32SetLocal(0)];
		let target = superoptimize_snippet(source_program, 0);
		assert_eq!(target, vec![]);
	}

	#[test]
	#[ignore] // TODO
	fn no_superoptimize_setlocal() {
		let source_program = &[I32Const(3), I32SetLocal(0)];
		let target = superoptimize_snippet(source_program, 1);
		assert_eq!(target, source_program);
	}
}
