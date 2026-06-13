import os
import requests

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_API_URL = os.getenv("WEATHER_API_URL")
POLLUTION_API_URL = os.getenv("POLLUTION_API_URL")
GEOCODING_API_URL = os.getenv("GEOCODING_API_URL")


@tool
def get_weather_details(location: str) -> str:
    """
    Get current weather for a city.
    Example input: Tokyo, Delhi, London
    """

    # Step 1: Convert city name to coordinates
    geo_response = requests.get(
        GEOCODING_API_URL,
        params={
            "q": location,
            "limit": 1,
            "appid": WEATHER_API_KEY,
        },
        timeout=10,
    )

    if geo_response.status_code != 200:
        return f"Failed to find location: {location}"

    geo_data = geo_response.json()

    if not geo_data:
        return f"Could not find coordinates for {location}"

    lat = geo_data[0]["lat"]
    lon = geo_data[0]["lon"]
    city = geo_data[0]["name"]
    country = geo_data[0]["country"]

    # Step 2: Get weather using coordinates
    weather_response = requests.get(
        WEATHER_API_URL,
        params={
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
            "units": "metric",
        },
        timeout=10,
    )

    if weather_response.status_code != 200:
        return f"Failed to fetch weather for {location}"

    weather_data = weather_response.json()

    temperature = weather_data["main"]["temp"]
    feels_like = weather_data["main"]["feels_like"]
    humidity = weather_data["main"]["humidity"]
    condition = weather_data["weather"][0]["description"]
    wind_speed = weather_data["wind"]["speed"]

    # Step 3: Get API details

    air_pollution_response = requests.get(
        POLLUTION_API_URL,
        params={
            "lat": lat,
            "lon": lon,
            "appid": WEATHER_API_KEY,
        },
        timeout=10,
    )

    if air_pollution_response.status_code != 200:
        return f"Failed to fetch air pollution data for {location}"
    
    air_pollution_data = air_pollution_response.json()

    api_index = air_pollution_data["list"][0]["main"]["aqi"]
    co_level = air_pollution_data["list"][0]["components"]["co"]
    no_level = air_pollution_data["list"][0]["components"]["no"]
    no2_level = air_pollution_data["list"][0]["components"]["no2"]
    o3_level = air_pollution_data["list"][0]["components"]["o3"]
    so2_level = air_pollution_data["list"][0]["components"]["so2"]
    pm2_5_level = air_pollution_data["list"][0]["components"]["pm2_5"]
    pm10_level = air_pollution_data["list"][0]["components"]["pm10"]
    nh3_level = air_pollution_data["list"][0]["components"]["nh3"]

    """ Only return what details are requested by the user. 
    For example, if the user only asks for temperature, then only return temperature. 
    If the user asks for all AQI details, then return AQI details. 
    If the user asks for all details, then return all details. """

    return (
        f"Weather in {city}, {country}\n"
        f"Condition: {condition}\n"
        f"Temperature: {temperature}°C\n"
        f"Feels Like: {feels_like}°C\n"
        f"Humidity: {humidity}%\n"
        f"Wind Speed: {wind_speed} m/s\n"
        f"Air Quality Index: {api_index}\n"
        f"Carbon Monoxide: {co_level} μg/m³\n"
        f"Nitrogen Oxide: {no_level} μg/m³\n"
        f"Nitrogen Dioxide: {no2_level} μg/m³\n"
        f"Ozone: {o3_level} μg/m³\n"
        f" sulfur Dioxide: {so2_level} μg/m³\n"
        f"Particulate Matter (2.5): {pm2_5_level} μg/m³\n"
        f"Particulate Matter (10): {pm10_level} μg/m³\n"
        f"Ammonia: {nh3_level} μg/m³\n"
    )


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

agent = create_agent(
    model=model,
    tools=[get_weather_details],
    system_prompt=(
        "You are a helpful weather assistant. "
        "Use the weather tool whenever weather information is requested."
    ),
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Tokyo?"
            }
        ]
    }
)

print(response["messages"][-1].content)