from langchain_core.tools import tool

# Basic tool
@tool
def get_Weather(city: str) -> str:
    """
    Get the current weather for a given city
    Use this when the user asks about weather, temperature or climate.
    
    Args:
        city: the name of the city (e.g. 'Mumbai', 'Delhi', 'London')
    Returns:
        The current weather for the given city
    """
    weather_data = {
        'mumbai': 'sunny',
        'delhi': 'cloudy',
        'london': 'rainy',
    }
    weather = weather_data.get(city.lower(), 'unknown')
    return f"The weather in {city} is {weather}"

@tool
def check_order(order_id: str) -> str:
    """Check order."""  # BAD: the LLM has no idea when to use this
    return "Order status: shipped"


@tool
def check_order_good(order_id: str) -> str:
    """
    Check the status of a customer order.

    Use this when the user asks where their order is, whether it has
    shipped, or when it will arrive.

    Args:
        order_id: The order number (e.g. 'ORD-12345' or just '12345')
    """
    return "Your order ORD-12345 has shipped. Expected delivery: 2 days."