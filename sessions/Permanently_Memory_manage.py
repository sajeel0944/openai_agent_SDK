import asyncio
from agents import SQLiteSession

session = SQLiteSession("2323", "Persistent_Memory.dp")


# Add some conversation items manually
async def add_conversation():
    conversation_items = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you?"},
        {"role": "user", "content": "What's the weather like?"},
        {"role": "assistant", "content": "I don't have access to weather data."}
    ]

    await session.add_items(conversation_items)
    print("Added conversation to memory!")


# View all items in memory
async def view_all_memory():
    items = await session.get_items()
    print(f"\nMemory contains {len(items)} items:")
    for item in items:
        print(f"  {item['role']}: {item['content']}")


# Remove the last item (undo)
async def remove_last_memory_item():
    last_item = await session.pop_item()
    print(f"\nRemoved last item: {last_item}")


# View memory again
async def view_memory_again():
    items = await session.get_items()
    print(f"\nMemory now contains {len(items)} items:")
    for item in items:
        print(f"  {item['role']}: {item['content']}")


# Clear all memory
async def clear_all_memory():
    await session.clear_session()
    print("\nCleared all memory!")

    # Verify memory is empty
    items = await session.get_items()
    print(f"Memory now contains {len(items)} items")



# Run the async demo
asyncio.run(add_conversation())
asyncio.run(view_all_memory())
asyncio.run(remove_last_memory_item())
asyncio.run(view_memory_again())
asyncio.run(clear_all_memory())
