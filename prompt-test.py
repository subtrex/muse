def flatten_list(nested_list):
    """Helper function to flatten a list of lists into a single list."""
    return [item for sublist in nested_list for item in sublist]

def classify_scene(categories, attributes):
    # Define music types, instruments, and audience impressions based on categories and attributes
    
    # Updated mappings for categories
    category_music_map = {
        "desert_road": ("Energetic", ["Guitar", "Drums"], "Excited"),
        "tundra": ("Epic", ["Synthesizer", "Drums"], "Motivated"),
        "mountain_path": ("Soothing", ["Piano", "Guitar"], "Relaxed"),
        "highway": ("Energetic", ["Guitar", "Drums"], "Excited"),
        "field_road": ("Soothing", ["Piano", "Flute"], "Relaxed"),
        "forest/broadleaf": ("Soothing", ["Piano", "Flute"], "Relaxed"),
        "rainforest": ("Soothing", ["Piano", "Synthesizer"], "Relaxed"),
        "forest_path": ("Soothing", ["Piano", "Flute"], "Relaxed"),
        "sky": ("Soothing", ["Piano", "Guitar"], "Relaxed"),
        "ski_slope": ("Energetic", ["Drums", "Guitar"], "Excited"),
        "outdoor": ("Energetic", ["Drums", "Guitar"], "Excited"),
        "campsite": ("Soothing", ["Guitar", "Drums"], "Relaxed"),
        "beach": ("Soothing", ["Guitar", "Flute"], "Relaxed"),
        "amusement_park": ("Energetic", ["Synthesizer", "Drums"], "Excited"),
        "city_street": ("Catchy", ["Guitar", "Drums"], "Excited"),
        "museum": ("Mellow", ["Piano", "Violin"], "Relaxed"),
        "library": ("Mellow", ["Piano", "Violin"], "Relaxed"),
        "hospital": ("Epic", ["Synthesizer", "Drums"], "Motivated"),
        "school": ("Inspirational", ["Piano", "Drums"], "Motivated"),
        "restaurant": ("Playful", ["Piano", "Trumpet"], "Happy"),
        "shopping_mall": ("Catchy", ["Synthesizer", "Drums"], "Excited"),
        "park": ("Soothing", ["Guitar", "Flute"], "Relaxed"),
        "gym": ("Energetic", ["Drums", "Bass Guitar"], "Excited"),
        "beach_house": ("Soothing", ["Guitar", "Flute"], "Relaxed"),
        "stadium": ("Energetic", ["Drums", "Guitar"], "Excited"),
        "cafe": ("Playful", ["Piano", "Ukulele"], "Happy"),
        "mountain": ("Epic", ["Strings", "Drums"], "Motivated"),
        "farm": ("Soothing", ["Guitar", "Flute"], "Relaxed"),
        "lake": ("Soothing", ["Piano", "Guitar"], "Relaxed"),
        "sunset": ("Romantic", ["Piano", "Violin"], "Nostalgic"),
        "sunrise": ("Soothing", ["Piano", "Flute"], "Relaxed"),
        "forest_clearing": ("Soothing", ["Guitar", "Flute"], "Relaxed"),
        "lake_shore": ("Soothing", ["Guitar", "Flute"], "Relaxed"),
        "busy_street": ("Catchy", ["Synthesizer", "Drums"], "Excited"),
        "quiet_street": ("Mellow", ["Piano", "Violin"], "Relaxed"),
        "sports_field": ("Energetic", ["Drums", "Guitar"], "Excited"),
        "public_square": ("Energetic", ["Drums", "Guitar"], "Excited"),
        "fireworks": ("Energetic", ["Synthesizer", "Drums"], "Excited"),
        "wedding": ("Romantic", ["Piano", "Violin"], "Nostalgic"),
        "party": ("Energetic", ["Synthesizer", "Drums"], "Excited"),
        "street_performance": ("Playful", ["Piano", "Guitar"], "Happy"),
        "dance_class": ("Energetic", ["Drums", "Synthesizer"], "Excited"),
        "meditation": ("Soothing", ["Flute", "Piano"], "Relaxed"),
        "nature_trail": ("Soothing", ["Guitar", "Flute"], "Relaxed"),
        "aquarium": ("Soothing", ["Piano", "Synthesizer"], "Relaxed"),
        "park_bench": ("Mellow", ["Piano", "Flute"], "Relaxed"),
        "art_gallery": ("Soothing", ["Piano", "Violin"], "Relaxed")
    }

    # Updated mappings for attributes
    attribute_music_map = {
        "boating": ("Soothing", ["Guitar", "Piano"], "Relaxed"),
        "driving": ("Energetic", ["Guitar", "Drums"], "Excited"),
        "biking": ("Energetic", ["Guitar", "Drums"], "Excited"),
        "camping": ("Soothing", ["Guitar", "Drums"], "Relaxed"),
        "reading": ("Mellow", ["Piano", "Violin"], "Relaxed"),
        "working": ("Inspirational", ["Piano", "Synthesizer"], "Motivated"),
        "socializing": ("Playful", ["Piano", "Trumpet"], "Happy"),
        "waiting_in_line": ("Soothing", ["Guitar", "Flute"], "Relaxed"),
        "sports": ("Energetic", ["Drums", "Guitar"], "Excited"),
        "shopping": ("Catchy", ["Synthesizer", "Drums"], "Excited"),
        "farming": ("Soothing", ["Guitar", "Flute"], "Relaxed"),
        "constructing": ("Energetic", ["Drums", "Guitar"], "Excited"),
        "praying": ("Soothing", ["Piano", "Flute"], "Relaxed"),
        "swimming": ("Soothing", ["Guitar", "Piano"], "Relaxed"),
        "diving": ("Energetic", ["Synthesizer", "Drums"], "Excited"),
        "picnic": ("Soothing", ["Guitar", "Flute"], "Relaxed"),
        "exercising": ("Energetic", ["Drums", "Bass Guitar"], "Excited"),
        "exploring": ("Soothing", ["Piano", "Guitar"], "Relaxed"),
        "movie_night": ("Mellow", ["Piano", "Violin"], "Relaxed"),
        "barbecue": ("Playful", ["Piano", "Guitar"], "Happy"),
        "road_trip": ("Energetic", ["Guitar", "Drums"], "Excited"),
        "celebration": ("Energetic", ["Synthesizer", "Drums"], "Excited"),
        "chill_out": ("Soothing", ["Guitar", "Piano"], "Relaxed"),
        "romantic_dinner": ("Romantic", ["Piano", "Violin"], "Nostalgic"),
        "relaxation": ("Soothing", ["Flute", "Piano"], "Relaxed"),
        "morning_routine": ("Soothing", ["Piano", "Flute"], "Relaxed")
    }

    # Default values
    default_music_type = "Soothing"
    default_instruments = ["Piano", "Drums"]
    default_impression = "Relaxed"

    # Extract and flatten the lists of categories and attributes
    top_categories = [item.split(" -> ")[1] for item in categories[:2]]  # Extract category names
    top_attributes = flatten_list(attributes)[:2]  # Flatten attributes and take top 2

    # Initialize values
    music_type = None
    instruments = []
    impression = None
    
    # Check categories first
    for cat in top_categories:
        if cat in category_music_map:
            music_type, instruments, impression = category_music_map[cat]
            break
    
    # If no valid category, try to combine attributes
    if not music_type:
        for attr in top_attributes:
            if attr in attribute_music_map:
                # If an attribute modifies the category's music type, use it
                attr_music_type, attr_instruments, attr_impression = attribute_music_map[attr]
                if not instruments:  # If instruments are not set, use attribute instruments
                    instruments = attr_instruments
                if not impression:  # If impression is not set, use attribute impression
                    impression = attr_impression
                if attr_music_type != default_music_type:  # Use attribute music type if different
                    music_type = attr_music_type
                else:
                    music_type = default_music_type
                break
    
    # Use default values if nothing else was set
    if not music_type:
        music_type = default_music_type
    if not instruments:
        instruments = default_instruments
    if not impression:
        impression = default_impression
    
    # Create the prompt
    prompt = f"Create a {music_type.lower()} music piece featuring {', '.join(instruments)} with {impression.lower()} feeling."
    return prompt

# Example usage
categories = ['0.207 -> desert_road', '0.119 -> tundra', '0.087 -> mountain_path', '0.087 -> highway', '0.041 -> field_road']
attributes = [['open area', 'natural light', 'far-away horizon', 'natural', 'rugged scene', 'clouds', 'hiking', 'dirt', 'climbing']]
prompt = classify_scene(categories, attributes)
print(prompt)