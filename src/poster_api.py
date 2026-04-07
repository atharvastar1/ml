import requests
import json
import os

# To use real posters, sign up for a free key at https://www.themoviedb.org/
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "") 

def get_movie_poster(movie_title, year=None):
    """
    Fetch movie poster URL from TMDB.
    Fallback to a high-quality placeholder if API key is missing or search fails.
    """
    # Remove year from title if present for better search (e.g., "Toy Story (1995)" -> "Toy Story")
    clean_title = movie_title.split("(")[0].strip()
    
    if TMDB_API_KEY:
        try:
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={clean_title}"
            response = requests.get(search_url).json()
            if response.get('results'):
                poster_path = response['results'][0].get('poster_path')
                if poster_path:
                    return f"https://image.tmdb.org/t/p/w500{poster_path}"
        except Exception as e:
            print(f"Error fetching from TMDB: {e}")
            
    # Fallback placeholders based on Unsplash (using movie-related keywords)
    # This ensures the UI always looks premium even without an API key.
    placeholders = [
        "https://images.unsplash.com/photo-1485846234645-a62644f84728?auto=format&fit=crop&q=80&w=500", # Cinema
        "https://images.unsplash.com/photo-1440404653325-ab127d49abc1?auto=format&fit=crop&q=80&w=500", # Film
        "https://images.unsplash.com/photo-1478720568477-152d9b164e26?auto=format&fit=crop&q=80&w=500", # Projector
        "https://images.unsplash.com/photo-1517604931442-7e0c8ed2963c?auto=format&fit=crop&q=80&w=500", # Theater
        "https://images.unsplash.com/photo-1536440136628-849c177e76a1?auto=format&fit=crop&q=80&w=500"  # Popcorn
    ]
    
    # Use title hash for consistent placeholder selection
    idx = hash(clean_title) % len(placeholders)
    return placeholders[idx]

if __name__ == "__main__":
    # Test
    print(get_movie_poster("Toy Story"))
