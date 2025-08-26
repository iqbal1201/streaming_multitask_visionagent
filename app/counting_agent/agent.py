import asyncio
from typing import AsyncGenerator

from google.adk.agents import LiveRequestQueue
from google.adk.agents.llm_agent import Agent
from google.adk.tools.function_tool import FunctionTool
from google.genai import Client
from google.genai import types as genai_types


# --- Stock Price Monitoring Tool ---
async def monitor_stock_price(stock_symbol: str):
    """Monitor the price for the given stock symbol continuously.

    Args:
        stock_symbol (str): The ticker symbol of the stock to monitor.
    Yields:
        str: Price updates for the stock.
    """
    print(f"Start monitoring stock price for {stock_symbol}!")

    # Mock stock price changes
    await asyncio.sleep(4)
    price_alert = f"The price for {stock_symbol} is 300"
    yield price_alert
    print(price_alert)

    await asyncio.sleep(4)
    price_alert = f"The price for {stock_symbol} is 400"
    yield price_alert
    print(price_alert)

    await asyncio.sleep(20)
    price_alert = f"The price for {stock_symbol} is 900"
    yield price_alert
    print(price_alert)

    await asyncio.sleep(20)
    price_alert = f"The price for {stock_symbol} is 500"
    yield price_alert
    print(price_alert)


# --- Video Stream Monitoring Tool ---
async def monitor_video_stream(input_stream: LiveRequestQueue):
    """Monitor how many mobile phone appears are in the video stream.

    Args:
        input_stream (LiveRequestQueue): Video input stream to be monitored.
    Yields:
        genai_types.GenerateContentResponse: Alerts when the number of mobile phone appearance.
    """
    print("start monitor_video_stream!")
    client = Client(vertexai=False)
    prompt_text = (
        "Count the number of mobile phone appears in this image. Do the calculation how many total mobile phones appear in the image. Just respond with a numeric number."
    )
    last_count = None

    while True:
        last_valid_req = None
        print("Start monitoring loop")

        # use this loop to pull the latest images and discard the old ones
        while input_stream._queue.qsize() != 0:
            live_req = await input_stream.get()
            if live_req.blob is not None and live_req.blob.mime_type == "image/jpeg":
                last_valid_req = live_req

        # If we found a valid image, process it
        if last_valid_req is not None:
            print("Processing the most recent frame from the queue")

            image_part = genai_types.Part.from_bytes(
                data=last_valid_req.blob.data,
                mime_type=last_valid_req.blob.mime_type
            )

            contents = genai_types.Content(
                role="user",
                parts=[image_part, genai_types.Part.from_text(prompt_text)],
            )

            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=(
                        "You are a helpful video analysis assistant. "
                        "You can count the number of mobile phone appears in this image or video. "
                        "Just respond with a numeric number."
                        "Do the calculation how many mobile phones appear in the image. Return with the total number."
                    )
                ),
            )

            new_count = response.candidates[0].content.parts[0].text
            if not last_count:
                last_count = new_count
            elif last_count != new_count:
                last_count = new_count
                yield response
                print("response:", response)

        await asyncio.sleep(0.5)


# --- Stop Streaming Tool ---
def stop_streaming(function_name: str):
    """Stop the streaming function.

    Args:
        function_name (str): The name of the streaming function to stop.
    """
    # Implementation depends on ADK agent lifecycle
    print(f"Stopping stream for function: {function_name}")


# --- Register tools ---
video_streaming_tool = FunctionTool(monitor_video_stream)
video_streaming_tool.name = "monitor_video_stream"
video_streaming_tool.description = "Monitor how many mobile phones appear in the video stream."
video_streaming_tool.__name__ = "monitor_video_stream"  # Patch for ADK compatibility

stock_price_tool = FunctionTool(monitor_stock_price)
stock_price_tool.name = "monitor_stock_price"
stock_price_tool.description = "Continuously monitor stock prices."
stock_price_tool.__name__ = "monitor_stock_price"  # Patch

stop_streaming_tool = FunctionTool(stop_streaming)
stop_streaming_tool.name = "stop_streaming"
stop_streaming_tool.description = "Stop a running streaming tool by function name."
stop_streaming_tool.__name__ = "stop_streaming"  # Patch


# --- Root Agent ---
root_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="video_streaming_agent",
    instruction="""
      You are a monitoring agent. You can do video monitoring and stock price monitoring
      using the provided tools/functions.
      When users want to monitor a video stream,
      you can use monitor_video_stream. When monitor_video_stream
      returns the alert, you should tell the users.
      When users want to monitor a stock price, you can use monitor_stock_price.
      If asked to stop a stream, call stop_streaming.
      Don't ask too many questions. Don't be too talkative.
    """,
    tools=[
        video_streaming_tool,
        stock_price_tool,
        stop_streaming_tool,
    ]
)
