#!/usr/bin/env python3
"""Test executor with async code and external functions."""

import asyncio

from code_sandboxes import CodeSandboxClient


async def main():
    # Create a sandbox
    client = CodeSandboxClient.create(variant="eval")
    client.start()

    # Set up an executor mock
    class MockExecutor:
        async def call_tool(self, name, args):
            print(f"[MockExecutor] Calling tool: {name} with {args}")
            await asyncio.sleep(0.01)
            return {"status": "success", "data": f"Result for {name}"}

    client.set_variable("__executor__", MockExecutor())

    # First: Define __call_tool__
    print("\n=== Step 1: Define __call_tool__ ===")
    result1 = client.execute_code("""
async def __call_tool__(tool_name, arguments):
    '''Call a tool through the executor.'''
    return await __executor__.call_tool(tool_name, arguments)

print("__call_tool__ defined successfully")
""")
    print(f"Result 1 - Error: {result1.code_error}")
    print(f"Result 1 - Stdout: {result1.stdout}")

    # Second: Use __call_tool__
    print("\n=== Step 2: Use __call_tool__ with await ===")
    result2 = client.execute_code("""
result = await __call_tool__("test_tool", {"arg": "value"})
print(f"Tool result: {result}")
result
""")
    print(f"Result 2 - Error: {result2.code_error}")
    print(f"Result 2 - Stdout: {result2.stdout}")
    print(f"Result 2 - Results: {result2.results}")

    client.stop()
    print("\n=== Test completed successfully! ===")


if __name__ == "__main__":
    asyncio.run(main())
