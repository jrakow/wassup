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
			let optimized = superoptimize_snippet(
				&vec,
				&function.local_types,
				ValueTypeConfig {
					i32_size: 4,
					i64_size: Some(8),
				},
			); // TODO
			*vec = optimized;
		}
	}

	*func_body = function.to_wasm_func_body();
}

pub fn superoptimize_snippet(
	source_program: &[Instruction],
	local_types: &[ValueType],
	value_type_config: ValueTypeConfig,
) -> Vec<Instruction> {
	let mut current_best = source_program.to_vec();

	loop {
		//		solver.push();
		let config = Config::default();
		let ctx = Context::new(&config);
		let solver = Solver::new(&ctx);

		// TODO compute actual stack types
		let initial_stack = vec![ValueType::I32; stack_depth(&source_program[..])];

		let mut initial_locals = Vec::new();
		let mut initial_locals_bounds = Vec::new();
		for ty in local_types {
			if *ty == ValueType::I32 {
				let sort = ctx.bitvector_sort(value_type_config.i32_size as u32);

				let bound = ctx.fresh_const("initial_local", &sort);
				initial_locals_bounds.push(bound.clone());

				initial_locals.push(value_type_config.i32_wrap_as_i64(&ctx, &bound));
			} else if *ty == ValueType::I64 {
				let sort = ctx.bitvector_sort(value_type_config.i64_size.unwrap() as u32);

				let bound = ctx.fresh_const("initial_local", &sort);
				initial_locals_bounds.push(bound.clone());

				initial_locals.push(bound)
			}
		}

		let constants = Constants::new(
			&ctx,
			&solver,
			initial_locals,
			local_types.to_vec(),
			&initial_stack,
			value_type_config,
		);

		let source_state = State::new(&ctx, &solver, &constants, "source_");
		let target_state = State::new(&ctx, &solver, &constants, "target_");
		source_state.set_source_program(&source_program[..]);

		let initial_locals_bounds: Vec<&Ast> = initial_locals_bounds.iter().collect();
		solver.assert(&ctx.forall_const(&initial_locals_bounds, &source_state.transitions()));
		solver.assert(&ctx.forall_const(&initial_locals_bounds, &target_state.transitions()));

		let target_length = &target_state.program_length();

		// force target program to be shorter than current best
		solver.assert(&in_range(
			&ctx.from_u64(0),
			target_length,
			&ctx.from_usize(current_best.len()),
		));

		// assert programs are equivalent for all local variables
		solver.assert(&ctx.forall_const(
			&initial_locals_bounds,
			&equivalent(
				&source_state,
				&ctx.from_usize(source_program.len()),
				&target_state,
				&target_length,
			),
		));

		if !solver.check() {
			// already optimal
			return current_best;
		}

		// better version found
		// decode

		let model = solver.get_model();
		current_best = target_state.decode_program(&model);

		//		solver.pop(1);
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use Instruction::*;
	use Value::*;

	#[test]
	#[ignore]
	fn superoptimize_nop() {
		let source_program = &[Const(I32(1)), Nop];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Const(I32(1))]);
	}

	#[test]
	#[ignore]
	fn superoptimize_consts_add() {
		let source_program = &[Const(I32(1)), Const(I32(2)), I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Const(I32(3))]);
	}

	#[test]
	#[ignore]
	fn superoptimize_consts_add_64bit() {
		let source_program = &[Const(I64(1)), Const(I64(2)), I64Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Const(I64(3))]);
	}

	#[test]
	#[ignore]
	fn superoptimize_add() {
		let source_program = &[Const(I32(0)), I32Add];
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![]);
	}

	#[test]
	fn no_superoptimize_setlocal() {
		let source_program = &[Const(I32(3)), SetLocal(0)];

		// no optimization possible, because locals cannot be changed
		let target = superoptimize_snippet(
			source_program,
			&[],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, source_program);
	}

	#[test]
	#[ignore] // TODO
	fn superoptimize_unreachable_garbage() {
		let source_program = &[Unreachable, GetLocal(0)];
		let target = superoptimize_snippet(
			source_program,
			&[ValueType::I32],
			ValueTypeConfig {
				i32_size: 32,
				i64_size: Some(64),
			},
		);
		assert_eq!(target, vec![Unreachable]);
	}
}
