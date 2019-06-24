#[cfg(test)]
mod integration {
	#[test]
	#[ignore]
	fn simple() {
		assert_cli::Assert::main_binary()
			.succeeds()
			.with_args(&["--stdin"])
			.stdin(
				r#"(module
				(func (result i32)
					i32.const 1
					nop
					i32.const 2
					i32.add
				)
			)"#,
			)
			.stdout()
			.is(r#"(module
  (type (;0;) (func (result i32)))
  (func (;0;) (type 0) (result i32)
    i32.const 3))"#)
			.unwrap();
	}
}
