use crate::instructions::*;
use crate::ValueTypeConfig;
use parity_wasm::elements::ValueType;
use z3::*;

/// Encoded types and expressions that are independent of a specific instruction sequence.
pub struct Constants<'ctx> {
	pub ctx: &'ctx Context,
	/// Values of the initial stack
	pub initial_stack: Vec<Ast<'ctx>>,
	/// Z3 constants used in initial_stack
	pub initial_stack_bounds: Vec<Ast<'ctx>>,
	pub initial_stack_types: Vec<Ast<'ctx>>,
	pub initial_locals: Vec<Ast<'ctx>>,
	pub n_locals: Ast<'ctx>,
	pub value_type_config: ValueTypeConfig,
}

impl<'ctx> Constants<'ctx> {
	pub fn new(
		ctx: &'ctx Context,
		solver: &Solver<'ctx>,
		initial_locals: Vec<Ast<'ctx>>,
		local_types: Vec<ValueType>,
		initial_stack_types: &[ValueType],
		value_type_config: ValueTypeConfig,
	) -> Self {
		let mut initial_stack = Vec::new();
		let mut initial_stack_bounds = Vec::new();
		let mut encoded_initial_stack_types = Vec::new();
		for ty in initial_stack_types {
			let encoded_value_type = value_type_config.encode_value_type(ctx, *ty);
			encoded_initial_stack_types.push(encoded_value_type);

			if *ty == ValueType::I32 {
				let sort = ctx.bitvector_sort(value_type_config.i32_size as u32);

				let bound = ctx.fresh_const("initial_stack", &sort);
				initial_stack_bounds.push(bound.clone());

				initial_stack.push(value_type_config.i32_wrap_as_i64(ctx, &bound));
			} else if *ty == ValueType::I64 {
				let sort = ctx.bitvector_sort(value_type_config.i64_size.unwrap() as u32);

				let bound = ctx.fresh_const("initial_stack", &sort);
				initial_stack_bounds.push(bound.clone());

				initial_stack.push(bound)
			}
		}

		let this = Self {
			ctx,
			initial_stack,
			initial_stack_bounds,
			initial_stack_types: encoded_initial_stack_types,
			initial_locals,
			n_locals: ctx.from_usize(local_types.len()),
			value_type_config,
		};

		// set local_type
		for (i, ty) in local_types.iter().enumerate() {
			let i = ctx.from_usize(i);
			let ty = value_type_config.encode_value_type(ctx, *ty);
			solver.assert(&this.local_type(&i)._eq(&ty));
		}

		this
	}

	/// How many values the instruction takes from the stack.
	///
	/// This is here so it can be encoded as a function, not as a big expression.
	pub fn stack_pop_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		let instruction_datatype = instruction_datatype(self.ctx, self.value_type_config);
		// build a switch statement from if-then-else
		let mut pop_switch = self.ctx.from_usize(0);
		for i in Instruction::iter_templates(self.value_type_config) {
			let (pops, _) = i.stack_pop_push_count();

			let variant = &instruction_datatype.variants[i.as_usize(self.value_type_config)];
			let active = variant.tester.apply(&[&instr]);

			pop_switch = active.ite(&self.ctx.from_usize(pops), &pop_switch);
		}
		pop_switch
	}

	/// How many values the instruction pushes on the stack.
	///
	/// This is here so it can be encoded as a function, not as a big expression.
	pub fn stack_push_count(&self, instr: &Ast<'ctx>) -> Ast<'ctx> {
		let instruction_datatype = instruction_datatype(self.ctx, self.value_type_config);

		// build a switch statement from if-then-else
		let mut push_switch = self.ctx.from_usize(0);
		for i in Instruction::iter_templates(self.value_type_config) {
			let (_, pushs) = i.stack_pop_push_count();

			let variant = &instruction_datatype.variants[i.as_usize(self.value_type_config)];
			let active = variant.tester.apply(&[&instr]);

			push_switch = active.ite(&self.ctx.from_usize(pushs), &push_switch);
		}
		push_switch
	}

	pub fn local_type(&self, index: &Ast<'ctx>) -> Ast<'ctx> {
		let local_type_func = self.ctx.func_decl(
			self.ctx.str_sym("local_type"),
			&[&self.ctx.int_sort()],
			&self.value_type_config.value_type_datatype(self.ctx).sort,
		);

		local_type_func.apply(&[index])
	}
}
