"""
https://docs.python.org/3/howto/a-conceptual-overview-of-asyncio.html#the-inner-workings-of-coroutines

asyncio leverages four components to pass around control.

1. coroutine.send(arg) is the method used to start or resume a coroutine. If the coroutine was paused and is now being resumed, the argument arg will be sent in as the return value of the yield statement which originally paused it. If the coroutine is being used for the first time (as opposed to being resumed), arg must be None.

2. yield, as usual, pauses execution and returns control to the caller. In the example above, the yield, on line 3, is called by ... = await rock on line 11. More broadly speaking, await calls the __await__() method of the given object. await also does one more very special thing: it propagates (or “passes along”) any yields it receives up the call chain. In this case, that’s back to ... = coroutine.send(None) on line 16.

3. The coroutine is resumed via the coroutine.send(42) call on line 21. The coroutine picks back up from where it yielded (or paused) on line 3 and executes the remaining statements in its body. 

4. When a coroutine finishes, it raises a StopIteration exception with the return value attached in the value attribute.

"""
class Rock:
    def __await__(self):
        value_sent_in = yield 7
        print(f"Rock.__await__ resuming with value: {value_sent_in}.")
        return value_sent_in

async def main():
    print("Beginning coroutine main().")
    rock = Rock()
    print("Awaiting rock...")
    value_from_rock = await rock
    print(f"Coroutine received value: {value_from_rock} from rock.")
    return 23

coroutine = main()
intermediate_result = coroutine.send(None)
print(f"Coroutine paused and returned intermediate value: {intermediate_result}.")

print(f"Resuming coroutine and sending in value: 42.")
try:
    coroutine.send(42)
except StopIteration as e:
    returned_value = e.value
print(f"Coroutine main() finished and provided value: {returned_value}.")