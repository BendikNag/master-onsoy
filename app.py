import streamlit as st
import pandas as pd
import numpy as np
import requests 
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Set page configuration
st.set_page_config(
    page_title="Masters Fantasy Tracker",
    page_icon="🏌️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("Masters Fantasy Tracker 2025 - Onsøy")

# Functions for data fetching and processing
def fetch_masters_scores(url=None):
    """
    Fetches the scores JSON data from the Masters tournament.
    If URL is None, uses the default URL.
    
    Returns:
        dict: The JSON data as a Python dictionary if successful, None otherwise.
    """
    if url is None:
        url = "https://www.masters.com/en_US/scores/feeds/2025/scores.json"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://www.masters.com/leaderboard"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def extract_specific_players(scores_data, player_ids):
    """
    Extracts information for specific player IDs from the scores data.
    
    Args:
        scores_data (dict): The complete scores data
        player_ids (list): List of player IDs to extract
    
    Returns:
        dict: A new dictionary containing only the specified players
    """
    if not scores_data or "data" not in scores_data or "player" not in scores_data["data"]:
        return {"error": "Invalid data format"}
    
    # Convert player_ids to a set for faster lookups
    player_ids_set = set(player_ids)
    
    # Filter players
    filtered_players = [player for player in scores_data["data"]["player"] 
                        if player["id"] in player_ids_set]
    
    # Create a new data structure with only the necessary information
    result = {
        "fileEpoch": scores_data["fileEpoch"],
        "data": {
            "currentRound": scores_data["data"]["currentRound"],
            "wallClockTime": scores_data["data"]["wallClockTime"],
            "statusRound": scores_data["data"]["statusRound"],
            "yardages": scores_data["data"]["yardages"],
            "pars": scores_data["data"]["pars"],
            "player": filtered_players
        }
    }
    
    return result

def get_player_information(player_id, data):
    """
    Extracts comprehensive player information from the data for a given player ID.
    
    Args:
        player_id (str): The ID of the player (as a string)
        data (dict): The scores data
    
    Returns:
        dict: A dictionary containing the player's information formatted as a player card
    """
    if not data or "data" not in data or "player" not in data["data"]:
        return {}
    
    # Find the player with the given ID
    for player in data["data"]["player"]:
        if player["id"] == str(player_id):  # Convert player_id to string for comparison
            # Extract rounds data
            rounds = []
            for i in range(1, 5):
                round_key = f"round{i}"
                if round_key in player:
                    round_data = player[round_key]
                    rounds.append({
                        "round_number": i,
                        "status": round_data["roundStatus"],
                        "tee_time": round_data["teetime"],
                        "total": round_data["total"],
                        "scores": round_data["scores"],
                        "fantasy_points": round_data.get("fantasy", 0)  # Default to 0 if fantasy not present
                    })
            
            return {
                "player_id": player["id"],
                "name": {
                    "full_name": player["full_name"],
                    "first_name": player["first_name"],
                    "last_name": player["last_name"],
                    "display_name": player["display_name"]
                },
                "country": {
                    "name": player["countryName"],
                    "code": player["countryCode"]
                },
                "tournament_status": {
                    "position": player["pos"],
                    "status": player["status"],
                    "active": player["active"],
                    "to_par": player["topar"],
                    "thru": player["thru"],
                    "today": player["today"],
                    "total_score": player["total"]
                },
                "player_attributes": {
                    "amateur": player["amateur"],
                    "first_timer": player["firsttimer"],
                    "past_champion": player["past"]
                },
                "tee_info": {
                    "current_tee_time": player["teetime"],
                    "group": player["group"],
                    "start": player["start"]
                },
                "rounds": rounds,
                "media": {
                    "has_image": player["image"],
                    "has_video": player["video"],
                    "last_highlight": player.get("last_highlight", "")
                }
            }
    
    return {}

def get_par_for_round(round_num, data):
    """Get par values for a specific round"""
    if not data or "data" not in data or "pars" not in data["data"]:
        return np.zeros(18)
    
    round_key = f"round{str(round_num)}"
    if round_key in data["data"]["pars"]:
        return np.array(data["data"]["pars"][round_key])
    
    # If no specific round data found, use the default pars (typically round1)
    if "round1" in data["data"]["pars"]:
        return np.array(data["data"]["pars"]["round1"])
    
    # Return default Augusta National pars if nothing else available
    return np.array([4, 5, 4, 3, 4, 3, 4, 5, 4, 4, 4, 3, 5, 4, 5, 3, 4, 4])

def get_round_score(round_num, player_id, data):
    """Get a player's score for a specific round"""
    if not data or "data" not in data or "player" not in data["data"]:
        return None
    
    player_data = data["data"]["player"]
    for player in player_data:
        if player["id"] == player_id:
            round_key = f"round{str(round_num)}"
            if round_key in player and "scores" in player[round_key]:
                # Handle the case where scores might contain None or non-numeric values
                scores = player[round_key]["scores"]
                # Convert to numpy array, handling None and non-numeric values
                scores_array = np.zeros(len(scores), dtype=object)
                for i, score in enumerate(scores):
                    if score is None or (isinstance(score, str) and not score.isdigit()):
                        scores_array[i] = None
                    else:
                        try:
                            scores_array[i] = int(score)
                        except (ValueError, TypeError):
                            scores_array[i] = None
                return scores_array
    
    return None

def get_fantasy_value_for_hole(strokes_to_par):
    """Calculate fantasy points for a single hole based on strokes to par"""
    if strokes_to_par == 0:  # Par
        return 1
    elif strokes_to_par == 1:  # Bogey
        return -1
    elif strokes_to_par >= 2:  # Double bogey or worse
        return -3
    elif strokes_to_par == -1:  # Birdie
        return 3
    elif strokes_to_par == -2:  # Eagle
        return 5
    elif strokes_to_par <= -3:  # Albatross or better
        return 8
    else:
        return 0

def check_no_bogey_bonus(strokes_to_par_list):
    """Check if a round qualifies for the bogey-free bonus"""
    # Filter out None values
    valid_scores = [s for s in strokes_to_par_list if s is not None]
    if not valid_scores:
        return False
    
    for strokes_to_par in valid_scores:
        if strokes_to_par >= 1:
            return False
    return True

def check_5_plus_birdies_or_better_bonus(strokes_to_par_list):
    """Check if a round has 5 or more birdies/eagles/etc."""
    # Filter out None values
    valid_scores = [s for s in strokes_to_par_list if s is not None]
    if not valid_scores:
        return False
    
    birdies_or_better = 0
    for strokes_to_par in valid_scores:
        if strokes_to_par <= -1:
            birdies_or_better += 1
    return birdies_or_better >= 5

def check_hole_in_one_bonus(strokes_list):
    """Count hole-in-ones in a round"""
    # Filter out None values
    valid_scores = [s for s in strokes_list if s is not None]
    if not valid_scores:
        return 0
    
    count_hole_in_one = 0
    for strokes in valid_scores:
        if strokes == 1:
            count_hole_in_one += 1
    return count_hole_in_one

def calc_fantasy_scores(player_ids, round_num, data):
    """Calculate fantasy scores for all players for a specific round"""
    score_overview = {}
    
    # Ensure player_ids is a flat list
    if isinstance(player_ids, dict):
        flat_ids = []
        for picker, ids in player_ids.items():
            if isinstance(ids, list):
                for player_id in ids:
                    flat_ids.append(player_id)
            else:
                flat_ids.append(ids)
        player_ids_list = flat_ids
    else:
        player_ids_list = player_ids
    
    for player_id in player_ids_list:
        # Skip if player_id is not a string
        if not isinstance(player_id, str):
            continue
            
        par = get_par_for_round(round_num, data)
        score = get_round_score(round_num, player_id, data)
        
        # Skip if no score data available
        if score is None or len(score) == 0:
            continue
            
        # Make sure par and score have the same length
        if len(par) != len(score):
            # Adjust or skip if lengths don't match
            if len(par) > len(score):
                par = par[:len(score)]
            else:
                score = score[:len(par)]
        
        # Calculate strokes to par, handling None values
        strokes_to_par_list = np.zeros_like(score, dtype=object)
        for i in range(len(score)):
            if score[i] is None:
                strokes_to_par_list[i] = None
            else:
                strokes_to_par_list[i] = score[i] - par[i]
        
        # Calculate hole-by-hole fantasy points
        hole_scores = 0
        for strokes_to_par in strokes_to_par_list:
            if strokes_to_par is not None:  # Skip holes not played yet
                hole_scores += get_fantasy_value_for_hole(strokes_to_par)
        
        # Calculate bonuses
        bogey_free_bonus = 5 if check_no_bogey_bonus(strokes_to_par_list) else 0
        birdie_bonus = 5 if check_5_plus_birdies_or_better_bonus(strokes_to_par_list) else 0
        hole_in_one_count = check_hole_in_one_bonus(score)
        hole_in_one_bonus = 10 * hole_in_one_count
        
        # Calculate total fantasy score
        total_fantasy_score = hole_scores + bogey_free_bonus + birdie_bonus + hole_in_one_bonus

        # Convert numpy arrays to lists for storage
        par_list = par.tolist()
        score_list = [s if s is not None else None for s in score]
        strokes_to_par_list_for_storage = [s if s is not None else None for s in strokes_to_par_list]

        # Store results
        score_overview[player_id] = {
            "round": round_num,
            "par": par_list,
            "score": score_list,
            "strokes_to_par": strokes_to_par_list_for_storage,
            "fantasy_score": total_fantasy_score,
            "hole_scores": hole_scores,
            "bogey_free": bool(bogey_free_bonus),
            "bogey_free_bonus": bogey_free_bonus,
            "five_plus_birdies": bool(birdie_bonus),
            "birdie_bonus": birdie_bonus,
            "hole_in_ones": hole_in_one_count,
            "hole_in_one_bonus": hole_in_one_bonus
        }
    
    return score_overview

def get_player_name_map(data):
    """Create a mapping from player ID to player name"""
    player_name_map = {}
    
    if not data or "data" not in data or "player" not in data["data"]:
        return player_name_map
    
    for player in data["data"]["player"]:
        player_name_map[player["id"]] = player["display_name"]
    
    return player_name_map

def get_fantasy_totals_by_picker(fantasy_pickers, wildcard_picks, all_rounds_scores, player_name_map):
    """Calculate total fantasy points for each fantasy picker and their players"""
    fantasy_totals = {}
    
    for picker_name, player_ids in fantasy_pickers.items():
        picker_total = 0
        player_details = {}
        
        # Original player scoring (all rounds)
        for player_id in player_ids:
            player_total = 0
            round_details = {}
            
            # Sum up points from all rounds
            for round_num, round_data in all_rounds_scores.items():
                if player_id in round_data:
                    points = round_data[player_id]["fantasy_score"]
                    player_total += points
                    round_details[round_num] = points
            
            player_name = player_name_map.get(player_id, f"Player {player_id}")
            player_details[player_name] = {
                "total": player_total,
                "rounds": round_details,
                "is_wildcard": False
            }
            picker_total += player_total
        
        # Wildcard player scoring (only rounds 3 and 4)
        if picker_name in wildcard_picks:
            for player_id in wildcard_picks[picker_name]:
                player_total = 0
                round_details = {}
                
                # Only include rounds 3 and 4 for wildcards
                for round_num in [3, 4]:
                    if round_num in all_rounds_scores and player_id in all_rounds_scores[round_num]:
                        points = all_rounds_scores[round_num][player_id]["fantasy_score"]
                        player_total += points
                        round_details[round_num] = points
                
                player_name = player_name_map.get(player_id, f"Player {player_id}")
                wildcard_label = f"{player_name} (WC)"
                player_details[wildcard_label] = {
                    "total": player_total,
                    "rounds": round_details,
                    "is_wildcard": True
                }
                picker_total += player_total
        
        fantasy_totals[picker_name] = {
            "total": picker_total,
            "players": player_details
        }
    
    return fantasy_totals

def get_current_round_from_data(data):
    """Extract the current round from the tournament data"""
    if not data or "data" not in data or "currentRound" not in data["data"]:
        return 1
    
    # Parse the current round and ensure it's an integer
    try:
        # Get the first digit of the current round
        current_round_str = str(data["data"]["currentRound"])
        for index in range(len(current_round_str)):
            if current_round_str[index] == '1':
                return index + 1
    except (ValueError, TypeError, IndexError):
        pass
    
    return 1

def format_to_par(value):
    """Format a score relative to par with + or - sign"""
    try:
        # Convert to integer if it's a string
        if isinstance(value, str):
            value = int(value)
        
        if value > 0:
            return f"+{value}"
        elif value < 0:
            return f"{value}"
        else:
            return "E"
    except (ValueError, TypeError):
        # Return the original value if conversion fails
        return str(value)

# Function to display player detail cards
def display_player_card(player_info, round_scores, tournament_status, is_wildcard=False):
    """Display a player's card with all their tournament information
    
    Args:
        player_info (dict): Player information dictionary
        round_scores (dict): Dictionary of round scores
        tournament_status (str): Tournament status
        is_wildcard (bool): Whether this player is a wildcard pick
    """
    if not player_info:
        return
    
    # Create a card-like container
    with st.container():
        # Add a wildcard badge if applicable
        if is_wildcard:
            st.subheader(f"{player_info['name']['full_name']} 🃏 (Wildcard)")
        else:
            st.subheader(player_info["name"]["full_name"])
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            st.markdown(f"**Current Position:** {player_info['tournament_status']['position']}")
            
            # Convert to_par to an integer if possible, otherwise use the string
            to_par = player_info['tournament_status']['to_par']
            st.markdown(f"**To Par:** {format_to_par(to_par)}")
            
            # Convert today's score to an integer if possible, otherwise use the string
            today = player_info['tournament_status']['today']
            st.markdown(f"**Today:** {format_to_par(today)}")
            
            st.markdown(f"**Thru:** {player_info['tournament_status']['thru']}")
            
            # Display country
            st.markdown(f"**Country:** {player_info['country']['name']}")
            
            # Player attributes
            attributes = []
            if player_info["player_attributes"]["amateur"]:
                attributes.append("Amateur")
            if player_info["player_attributes"]["first_timer"]:
                attributes.append("First Timer")
            if player_info["player_attributes"]["past_champion"]:
                attributes.append("Past Champion")
            
            if attributes:
                st.markdown(f"**Attributes:** {', '.join(attributes)}")
        
        with col2:
            st.markdown("### Round Scores")
            for round_data in player_info["rounds"]:
                round_num = round_data["round_number"]
                round_status = round_data["status"]
                round_total = round_data["total"]
                
                # Skip rounds 1-2 for wildcards
                if is_wildcard and round_num < 3:
                    continue
                
                st.markdown(f"**Round {round_num}:** {round_total} ({round_status})")
        
        with col3:
            st.markdown("### Fantasy Performance")
            
            fantasy_data = []
            for round_num in range(1, 5):
                # Skip rounds 1-2 for wildcards
                if is_wildcard and round_num < 3:
                    continue
                    
                player_id = player_info["player_id"]
                if player_id in round_scores.get(round_num, {}):
                    round_fantasy = round_scores[round_num][player_id]
                    
                    fantasy_data.append({
                        "Round": f"Round {round_num}",
                        "Fantasy Points": round_fantasy["fantasy_score"],
                        "Hole Points": round_fantasy["hole_scores"],
                        "Bonuses": round_fantasy["bogey_free_bonus"] + round_fantasy["birdie_bonus"] + round_fantasy["hole_in_one_bonus"]
                    })
            
            if fantasy_data:
                fantasy_df = pd.DataFrame(fantasy_data)
                st.dataframe(fantasy_df, hide_index=True)
                
                # Calculate and display total fantasy points
                total_fantasy = sum(row["Fantasy Points"] for row in fantasy_data)
                st.markdown(f"**Total Fantasy Points:** {total_fantasy}")
                
                # Display bonus information
                bonuses = []
                for round_num in range(1, 5):
                    # Skip rounds 1-2 for wildcards
                    if is_wildcard and round_num < 3:
                        continue
                        
                    player_id = player_info["player_id"]
                    if player_id in round_scores.get(round_num, {}):
                        round_fantasy = round_scores[round_num][player_id]
                        if round_fantasy["bogey_free"]:
                            bonuses.append(f"Bogey-free round in Round {round_num} (+5 pts)")
                        if round_fantasy["five_plus_birdies"]:
                            bonuses.append(f"5+ birdies in Round {round_num} (+5 pts)")
                        if round_fantasy["hole_in_ones"] > 0:
                            bonuses.append(f"{round_fantasy['hole_in_ones']} hole-in-one in Round {round_num} (+10 pts each)")
                
                if bonuses:
                    st.markdown("### Bonuses Earned")
                    for bonus in bonuses:
                        st.markdown(f"- {bonus}")
        
        # Display detailed hole-by-hole scores
        st.markdown("### Hole-by-Hole Scores")
        
        # Get all rounds data for this player
        all_rounds_data = []
        player_id = player_info["player_id"]
        
        for round_num in range(1, 5):
            # Skip rounds 1-2 for wildcards
            if is_wildcard and round_num < 3:
                continue
                
            if player_id in round_scores.get(round_num, {}):
                round_data = round_scores[round_num][player_id]
                
                # Get scores and pars for this round
                scores = round_data["score"]
                pars = round_data["par"]
                
                all_rounds_data.append({
                    "round_num": round_num, 
                    "scores": scores, 
                    "pars": pars
                })
        
        if all_rounds_data:
            # For each round, create a simple two-row table
            for round_data in all_rounds_data:
                round_num = round_data["round_num"]
                scores = round_data["scores"]
                pars = round_data["pars"]
                
                # Display round number
                st.subheader(f"Round {round_num}")
                
                # Create a simple table with two rows - hole numbers and scores
                holes = ["Hole"] + [str(i) for i in range(1, 19)] + ["Total"]
                par_row = ["Par"]
                score_row = ["Score"]
                
                # Fill in par values
                total_par = 0
                for i in range(18):
                    if i < len(pars) and pars[i] is not None:
                        par_val = int(pars[i])
                        total_par += par_val
                        par_row.append(str(par_val))
                    else:
                        par_row.append("-")
                
                # Fill in score values
                total_score = 0
                valid_scores = 0
                for i in range(18):
                    if i < len(scores) and scores[i] is not None:
                        score_val = int(scores[i])
                        total_score += score_val
                        valid_scores += 1
                        
                        # Determine if this is under or over par
                        if i < len(pars) and pars[i] is not None:
                            par_val = int(pars[i])
                            diff = score_val - par_val
                            
                            # Format score with indicator
                            if score_val == 1:  # Hole in one
                                score_row.append(f"{score_val} 🥇")
                            elif diff <= -3:  # Albatross or better
                                score_row.append(f"{score_val} 🤯")
                            elif diff == -2:  # Eagle
                                score_row.append(f"{score_val} 🦅")
                            elif diff == -1:  # Birdie
                                score_row.append(f"{score_val} 🐥")
                            elif diff == 0:  # Par
                                score_row.append(str(score_val))
                            elif diff == 1:  # Bogey
                                score_row.append(f"{score_val} 🔴")
                            elif diff >= 2:  # Double bogey or worse
                                score_row.append(f"{score_val} 💩")
                        else:
                            score_row.append(str(score_val))
                    else:
                        score_row.append("-")
                
                # Add totals
                par_row.append(str(total_par))
                
                if valid_scores > 0:
                    # Add total with indicator of overall performance
                    if total_score < total_par:
                        score_row.append(f"{total_score} ({format_to_par(total_score - total_par)})")
                    elif total_score > total_par:
                        score_row.append(f"{total_score} ({format_to_par(total_score - total_par)})")
                    else:
                        score_row.append(f"{total_score} (E)")
                else:
                    score_row.append("-")
                
                # Create and display the table
                round_table = pd.DataFrame([par_row, score_row], columns=holes)
                st.table(round_table)
                
                st.markdown("---")
        
        # Horizontal line to separate players
        st.markdown("---")

# Function to create a round-by-round leaderboard
def create_round_leaderboard(fantasy_pickers, wildcard_picks, round_scores, player_name_map, round_num):
    """Create a leaderboard DataFrame for a specific round"""
    leaderboard_data = []
    
    for picker_name, player_ids in fantasy_pickers.items():
        picker_total = 0
        player_scores = []
        
        # Regular players
        for player_id in player_ids:
            if player_id in round_scores:
                player_score = round_scores[player_id]["fantasy_score"]
                picker_total += player_score
                
                player_name = player_name_map.get(player_id, f"Player {player_id}")
                player_scores.append((player_name, player_score, False))
        
        # Wildcard players (only for rounds 3-4)
        if round_num >= 3 and picker_name in wildcard_picks:
            for player_id in wildcard_picks[picker_name]:
                if player_id in round_scores:
                    player_score = round_scores[player_id]["fantasy_score"]
                    picker_total += player_score
                    
                    player_name = player_name_map.get(player_id, f"Player {player_id}")
                    player_scores.append((f"{player_name}", player_score, True))
        
        leaderboard_data.append({
            "Fantasy Picker": picker_name,
            "Total Points": picker_total,
            "Player Details": player_scores
        })
    
    # Sort by total points (descending)
    leaderboard_df = pd.DataFrame(leaderboard_data).sort_values(by="Total Points", ascending=False)
    leaderboard_df = leaderboard_df.reset_index(drop=True)
    
    # Add rank column
    leaderboard_df.index = leaderboard_df.index + 1
    leaderboard_df = leaderboard_df.rename_axis("Rank")
    
    return leaderboard_df

# Function to create the all-rounds cumulative leaderboard
def create_cumulative_leaderboard(fantasy_totals):
    """Create a leaderboard DataFrame with cumulative scores"""
    leaderboard_data = []
    
    for picker_name, picker_data in fantasy_totals.items():
        picker_total = picker_data["total"]
        player_details = picker_data["players"]
        
        leaderboard_data.append({
            "Fantasy Picker": picker_name,
            "Total Points": picker_total,
            "Player Details": player_details
        })
    
    # Sort by total points (descending)
    leaderboard_df = pd.DataFrame(leaderboard_data).sort_values(by="Total Points", ascending=False)
    leaderboard_df = leaderboard_df.reset_index(drop=True)
    
    # Add rank column
    leaderboard_df.index = leaderboard_df.index + 1
    leaderboard_df = leaderboard_df.rename_axis("Rank")
    
    return leaderboard_df

# Define fantasy pickers and their player selections
fantasy_pickers = {
    "Fredrik": ["46046", "33204", "46414", "37378"],
    "Viktor": ["46970", "47959", "30911", "40250"],
    "Bendik": ["50525", "46442", "52215", "34046"],
    "Thomas": ["21209", "28237", "58168", "52955"],
    "Haugstvedt": ["30925", "34363", "52686", "48081"],
    "BH": ["32839", "49960", "33653", "33448"],
    "Aksel": ["34360", "46717", "45523", "36689"]
}

# Define wildcard picks for rounds 3-4
wildcard_picks = {
    "Fredrik": ["47995"],  # Add player IDs for Fredrik's wildcards
    "Viktor": ["59141"],   # Add player IDs for Viktor's wildcards
    "Bendik": ["40098"],   # Add player IDs for Bendik's wildcards
    "Thomas": ["26331"],   # Add player IDs for Thomas's wildcards
    "Haugstvedt": ["25804"],  # Add player IDs for Haugstvedt's wildcards
    "BH": ["22405"],      # Add player IDs for BH's wildcards
    "Aksel": ["39997", "51349"]     # Add player IDs for Aksel's wildcards
}

# Flatten player IDs for data fetching (include wildcards)
player_ids_flatten = []
for picker, ids in fantasy_pickers.items():
    for player_id in ids:
        player_ids_flatten.append(player_id)

# Add wildcard players to the flattened list
for picker, ids in wildcard_picks.items():
    for player_id in ids:
        if player_id not in player_ids_flatten:
            player_ids_flatten.append(player_id)

# Add a function to create a cache key based on current time
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_cached_data():
    """Fetch and cache the Masters data"""
    scores_data = fetch_masters_scores()
    if not scores_data:
        st.error("Failed to get data from API")
        return None
    
    filtered_data = extract_specific_players(scores_data, player_ids_flatten)
    return filtered_data

# Main application code
try:
    # Get cached data
    with st.spinner("Fetching tournament data..."):
        data = get_cached_data()
    
    if data is None:
        st.error("Failed to fetch tournament data. Please try again later.")
        st.stop()
    
    # Get the current round
    current_round = get_current_round_from_data(data)
    
    # Calculate fantasy scores for all rounds
    all_rounds_scores = {}
    for round_num in range(1, current_round + 1):
        all_rounds_scores[round_num] = calc_fantasy_scores(player_ids_flatten, round_num, data)
    
    # Create player name map
    player_name_map = get_player_name_map(data)
    
    # Calculate fantasy totals by picker
    fantasy_totals = get_fantasy_totals_by_picker(fantasy_pickers, wildcard_picks, all_rounds_scores, player_name_map)
    
    # Create sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Overview", "Team Details", "Wildcards"])
    
    # Display the selected page
    if page == "Overview":
        st.header("Tournament Overview")
        
        # Add refresh button
        if st.button("Refresh Data", key="refresh_button"):
            st.cache_data.clear()
            st.rerun()
        
        # Tournament status information
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Round", str(current_round))
        with col2:
            # Format the wall clock time
            wall_clock_time = data["data"]["wallClockTime"]
            try:
                # Try to parse and format the time
                timestamp = int(wall_clock_time) / 1000  # Convert milliseconds to seconds
                formatted_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                st.metric("Last Updated", formatted_time)
            except (ValueError, TypeError):
                st.metric("Last Updated", wall_clock_time)
        
        # Current Round Leaderboard
        st.subheader(f"Round {current_round} Leaderboard")
        current_round_leaderboard = create_round_leaderboard(
            fantasy_pickers, 
            wildcard_picks if current_round >= 3 else {},  # Only include wildcards in rounds 3-4
            all_rounds_scores.get(current_round, {}), 
            player_name_map,
            current_round
        )
        
        # Display the leaderboard
        st.dataframe(current_round_leaderboard[["Fantasy Picker", "Total Points"]], hide_index=False)
        
        # Show the detailed breakdown for the current round
        st.subheader(f"Round {current_round} Detailed Breakdown")
        for index, row in current_round_leaderboard.iterrows():
            picker = row["Fantasy Picker"]
            total = row["Total Points"]
            player_details = row["Player Details"]
            
            # Player details for this picker
            st.markdown(f"**{picker}**: {total} points")
            
            # Create a formatted list of player scores
            for player_name, score, is_wildcard in player_details:
                wildcard_tag = " (Wildcard)" if is_wildcard else ""
                st.markdown(f"- {player_name}{wildcard_tag}: {score} points")
            
            st.markdown("---")
        
        # Cumulative Leaderboard
        st.header("Tournament Cumulative Leaderboard")
        cumulative_leaderboard = create_cumulative_leaderboard(fantasy_totals)
        
        # Display the cumulative leaderboard
        st.dataframe(cumulative_leaderboard[["Fantasy Picker", "Total Points"]], hide_index=False)
        
        # Detailed breakdown of all fantasy pickers
        st.subheader("Cumulative Fantasy Points Breakdown")
        for picker_name, picker_data in fantasy_totals.items():
            with st.expander(f"{picker_name}: {picker_data['total']} points"):
                for player_name, player_data in picker_data["players"].items():
                    player_total = player_data["total"]
                    rounds = player_data["rounds"]
                    is_wildcard = player_data["is_wildcard"]
                    
                    wildcard_tag = " (Wildcard - Rounds 3-4 only)" if is_wildcard else ""
                    round_details = ", ".join([f"R{r}: {pts}" for r, pts in rounds.items()])
                    st.markdown(f"**{player_name}{wildcard_tag}**: {player_total} points ({round_details})")
                    
    elif page == "Team Details":
        st.header("Team Details")
        
        # Team selector
        selected_team = st.selectbox(
            "Select Fantasy Picker",
            list(fantasy_pickers.keys())
        )
        
        # Display the selected team's information
        if selected_team:
            st.subheader(f"{selected_team}'s Team")
            
            # Get the team's player IDs
            player_ids = fantasy_pickers[selected_team]
            
            # Get the wildcard player IDs if available
            wildcard_ids = wildcard_picks.get(selected_team, [])
            
            # Get and display total team points
            team_total = fantasy_totals[selected_team]["total"]
            st.markdown(f"### Total Team Points: {team_total}")
            
            # Create a bar chart showing points by player
            player_points = {}
            wildcard_status = {}
            for player_name, player_data in fantasy_totals[selected_team]["players"].items():
                player_points[player_name] = player_data["total"]
                wildcard_status[player_name] = player_data["is_wildcard"]
            
            player_df = pd.DataFrame({
                "Player": list(player_points.keys()),
                "Fantasy Points": list(player_points.values()),
                "Is Wildcard": [wildcard_status[name] for name in player_points.keys()]
            })
            
            # Sort by points (descending)
            player_df = player_df.sort_values(by="Fantasy Points", ascending=False)
            
            # Create a bar chart with color distinction for wildcards
            fig = px.bar(
                player_df, 
                x="Player", 
                y="Fantasy Points", 
                title=f"{selected_team}'s Players Performance",
                color="Is Wildcard",
                color_discrete_map={True: "gold", False: "blue"},
                labels={"Is Wildcard": "Wildcard Pick"}
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display detailed player information for regular picks
            st.subheader("Regular Team")
            
            # For each player in the team
            for player_id in player_ids:
                # Get player information
                player_info = get_player_information(player_id, data)
                
                # Display player card
                display_player_card(player_info, all_rounds_scores, data["data"]["statusRound"])
            
            # Display detailed player information for wildcard picks
            if wildcard_ids:
                st.subheader("Wildcard Picks (Rounds 3-4 Only)")
                
                # For each wildcard player
                for player_id in wildcard_ids:
                    # Get player information
                    player_info = get_player_information(player_id, data)
                    
                    # Display player card with wildcard flag
                    display_player_card(player_info, all_rounds_scores, data["data"]["statusRound"], is_wildcard=True)
            
            # Show emoji legend
            st.markdown("""
            **Score Indicators:**  
            🐥 = Birdie\n
            🦅 = Eagle\n
            🤯 = Albatross or better\n
            🥇 = Hole in One\n
            🔴 = Bogey\n
            💩 = Double bogey or worse
            """)
            
    elif page == "Wildcards":
        st.header("Wildcard Picks")
        
        st.markdown("""
        ### Wildcard Rules
        - Each fantasy picker can select two wildcard players for Rounds 3 and 4
        - Wildcard players only score points for Rounds 3 and 4
        """)
        
        # Create a table showing all wildcard selections
        wildcard_data = []
        for picker_name, picker_wildcards in wildcard_picks.items():
            player_names = [player_name_map.get(player_id, f"Player {player_id}") for player_id in picker_wildcards]
            
            # Calculate points from wildcard picks
            wildcard_points = 0
            if picker_name in fantasy_totals:
                for player_name, player_data in fantasy_totals[picker_name]["players"].items():
                    if player_data["is_wildcard"]:
                        wildcard_points += player_data["total"]
            
            wildcard_data.append({
                "Fantasy Picker": picker_name,
                "Wildcard Picks": ", ".join(player_names),
                "Wildcard Points": wildcard_points
            })
        
        # Sort by wildcard points (descending)
        wildcard_df = pd.DataFrame(wildcard_data).sort_values(by="Wildcard Points", ascending=False)
        
        # Display the wildcard table
        st.dataframe(wildcard_df, hide_index=True)
        
        # Create a bar chart for wildcard performance
        st.subheader("Wildcard Performance by Fantasy Picker")
        
        fig = px.bar(
            wildcard_df,
            x="Fantasy Picker",
            y="Wildcard Points",
            title="Points From Wildcard Picks (Rounds 3-4)",
            color="Wildcard Points",
            color_continuous_scale="Viridis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed breakdown of individual wildcard players
        st.subheader("All Wildcard Picks")
        
        # Get unique wildcard player IDs
        unique_wildcard_players = set()
        for wildcards in wildcard_picks.values():
            for player_id in wildcards:
                unique_wildcard_players.add(player_id)
        
        # Calculate total points for each wildcard player
        wildcard_player_points = {}
        for player_id in unique_wildcard_players:
            player_name = player_name_map.get(player_id, f"Player {player_id}")
            
            # Calculate points for rounds 3-4
            total_points = 0
            for round_num in [3, 4]:
                if round_num in all_rounds_scores and player_id in all_rounds_scores[round_num]:
                    total_points += all_rounds_scores[round_num][player_id]["fantasy_score"]
            
            wildcard_player_points[player_name] = total_points
        
        # Create a dataframe and display
        wildcard_player_df = pd.DataFrame({
            "Player": list(wildcard_player_points.keys()),
            "Fantasy Points (Rounds 3-4)": list(wildcard_player_points.values())
        }).sort_values(by="Fantasy Points (Rounds 3-4)", ascending=False)
        
        st.dataframe(wildcard_player_df, hide_index=True)
        
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)