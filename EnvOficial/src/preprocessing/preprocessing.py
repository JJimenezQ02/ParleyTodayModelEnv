import pandas as pd 
import numpy as np 
from typing import List, Union, Dict, Tuple
from pathlib import Path


def preprocessing_1x2_extra_time_filter(df: pd.DataFrame, data_path: Path, lower_threshold: int = -30, upper_threshold: int = 30) -> pd.DataFrame:
    tms_match_stats = pd.read_parquet(data_path / 'team_match_stats.parquet')
    # Inspect
    # tms_match_stats.head()
    # Inspect
    # tms_match_stats['period'].value_counts()
    # Filtering
    matches_with_extra_time = (
        tms_match_stats.loc[
            (tms_match_stats["period"] == "ET1") &
            (tms_match_stats["possession_percentage"].notna()),
            "match_id"
        ]
        .unique()
    )

    DOUBLE_LEG_CROSS_TOURNAMENT = [
        "uefa-europa-league",
        "uefa-champions-league",
        "conference-league",
        "copa_de_italia",
        "copa-del-rey",
        "carabao-cup",
    ]


    df_filtered = df[
        ~(
            df["match_id"].isin(matches_with_extra_time) &
            df["tournament"].isin(DOUBLE_LEG_CROSS_TOURNAMENT)
        )
    ]
    print(f"The Number of matches eliminated on this filter is: {df.shape[0] - df_filtered.shape[0]} & Total: {df_filtered.shape[0]}")
    ### Winsor rest_days since some teams might have few apperances in the df
    df_filtered["diff_days_rest"] = df_filtered["diff_days_rest"].clip(lower=lower_threshold, upper=upper_threshold)

    return df_filtered


def clean_dataset(df: pd.DataFrame) -> Tuple[list[str], list[str], list[str]]:
    TARGET_COL = ['target_result']
    EXCLUDE_LIST = [
        'generated_at', '1x2_confidence', '1x2_prediction', '1x2_prob_away', '1x2_prob_draw',
        '1x2_prob_home', 'away_aerial_duel_win_rate', 'away_cards_per_foul', 'away_corners_for', 'away_opponent_tackle_success_rate',
        'away_opponent_tackles_successful', 'away_season_gd_per_game', 'away_season_ppg', 'away_tackle_failure_rate', 'away_team_name',
        'away_team_id', 'diff_aerial_duel_win_rate', 'diff_aerial_duels', 'diff_buildup_score', 'diff_cards_per_foul',
        'diff_clean_sheet', 'diff_clearances', 'diff_corners_against', 'diff_corners_for', 'diff_crosses_total',
        'diff_dispossessed', 'diff_dribbles_successful', 'diff_expected_goals', 'diff_fouled_in_final_third', 'diff_fouls_committed',
        'diff_fouls_drawn', 'diff_fouls_padj', 'diff_free_kicks', 'diff_goal_kicks', 'diff_goals_conceded',
        'diff_goals_scored', 'diff_is_home_team', 'diff_is_win', 'diff_match_pace_index', 'diff_match_points',
        'diff_opponent_crosses', 'diff_opponent_dispossessed', 'diff_opponent_dribbles_successful', 'diff_opponent_fouled_in_final_third', 'diff_opponent_free_kicks',
        'diff_opponent_goal_kicks', 'diff_opponent_offsides', 'diff_opponent_red_cards', 'diff_opponent_sot', 'diff_opponent_tackle_success_rate',
        'diff_opponent_tackles', 'diff_opponent_tackles_successful', 'diff_opponent_touches_in_box', 'diff_opponent_yellow_cards', 'diff_possession_pct',
        'diff_ppda', 'diff_recoveries', 'diff_red_cards', 'diff_ref_team_win_pct_smoothed', 'diff_score',
        'diff_season_gd_per_game', 'diff_season_goal_diff', 'diff_season_id', 'diff_season_id_right', 'diff_season_points',
        'diff_season_ppg', 'diff_season_progression', 'diff_shots_faced', 'diff_shots_taken', 'diff_sot',
        'diff_tackle_failure_rate', 'diff_tackles_successful', 'diff_tackles_total', 'diff_team_id', 'diff_total_duels',
        'diff_touches_in_box', 'diff_tournament_id', 'diff_xg_padj', 'diff_xga', 'diff_xi_avg_finishing_overperformance_per_90',
        'diff_xi_avg_recent_match_fitness', 'diff_xi_avg_season_fouls_committed_per_90', 'diff_xi_avg_season_goals_per_game', 'diff_xi_avg_season_red_cards_per_90', 'diff_xi_avg_season_xg_per_game',
        'diff_xi_avg_season_yellow_cards_per_90', 'diff_xi_avg_squad_season_progression', 'diff_yellow_cards', 'home_cards_per_foul', 'home_corners_for',
        'home_opponent_tackle_success_rate', 'home_score', 'home_season_gd_per_game', 'home_season_ppg', 'home_tackle_failure_rate',
        'home_team_id', 'home_team_name', 'home_touches_in_box', 'home_xi_avg_finishing_overperformance_per_90', 'home_xi_avg_season_fouls_committed_per_90',
        'home_xi_avg_season_goals_per_game', 'home_xi_avg_season_xg_per_game', 'home_xi_avg_season_yellow_cards_per_90', 'home_yellow_cards', 'market_max_prob',
        'match_datetime_utc', 'odds_prob_away', 'odds_prob_draw', 'odds_prob_home', 'season_id',
        'target_btts', 'target_result', 'target_total_goals', 'tournament_id', 'home_season_name',
        'home_country_name', 'home_season_id', 'home_tournament_name', 'home_is_home_team', 'home_match_datetime_utc_right',
        'away_season_name', 'away_country_name', 'away_match_datetime_utc_right', 'away_tournament_name', 'match_id',
        'away_season_id', 'ref_career_games_count', 'home_season_games_played', 'away_season_games_played', 'buildup_score',
        'home_team_avg_buildup_last_5', 'home_team_avg_buildup_last_10', 'away_team_avg_buildup_last_5', 'away_team_avg_buildup_last_10', 'home_ref_team_history_matches',
        'away_ref_team_history_matches', 'home_venue_avg_attendance_pct_last_5', 'home_season_id_right', 'home_tournament_id', 'away_name',
        'away_season_id_right', 'home_name', 'away_tournament_id', 'ref_cards_per_foul_last_5', 'home_xi_sum_season_yellow_cards',
        'home_season_points', 'away_season_points', 'home_tournament', 'away_tournament', 'name',
        'home_corners_against', 'home_shots_taken', 'home_shots_faced', 'home_match_pace_index', 'home_possession_pct',
        'home_fouls_committed', 'home_fouls_drawn', 'away_is_home_team', 'home_is_home_team', 'away_corners_against',
        'away_shots_taken', 'away_shots_faced', 'away_match_pace_index', 'away_possession_pct', 'away_fouls_committed',
        'away_fouls_drawn', 'home_red_cards', 'home_opponent_yellow_cards', 'home_opponent_red_cards', 'home_tackles_total',
        'home_tackles_successful', 'home_opponent_tackles', 'home_opponent_tackles_successful', 'home_opponent_dribbles_successful', 'home_dribbles_successful',
        'home_total_duels', 'home_aerial_duels', 'home_aerial_duel_win_rate', 'home_dispossessed', 'home_opponent_dispossessed',
        'home_free_kicks', 'home_opponent_free_kicks', 'home_sot', 'home_opponent_sot', 'home_clearances',
        'home_recoveries', 'home_opponent_touches_in_box', 'home_opponent_crosses', 'home_crosses_total', 'home_offsides',
        'home_opponent_offsides', 'home_goal_kicks', 'home_opponent_goal_kicks', 'away_yellow_cards', 'away_red_cards',
        'away_opponent_yellow_cards', 'away_opponent_red_cards', 'away_tackles_total', 'away_tackles_successful', 'away_opponent_tackles',
        'away_opponent_tackles_successful', 'away_opponent_dribbles_successful', 'away_dribbles_successful', 'away_total_duels', 'away_aerial_duels',
        'away_free_kicks', 'away_opponent_free_kicks', 'away_sot', 'away_opponent_sot', 'away_clearances',
        'away_recoveries', 'away_opponent_touches_in_box', 'away_touches_in_box', 'away_opponent_crosses', 'away_crosses_total',
        'away_offsides', 'away_opponent_offsides', 'away_goal_kicks', 'away_opponent_goal_kicks', 'total_corners', 'diff_offsides',
        'home_total_corners', 'away_total_corners', 'home_fouls_padj', 'away_fouls_padj', 'home_total_total_shots', 'away_total_total_shots',
        #COMPLEXITY
        'away_schedule_next_game_is_diff_tournament', 'away_schedule_next_game_is_high_priority',
        'away_schedule_next_opp_elo_vs_league',  'diff_schedule_days_until_next',
        'diff_schedule_next_game_is_diff_tournament', 'diff_schedule_next_game_is_high_priority',
        'diff_schedule_next_opp_elo_vs_league', 'home_schedule_next_game_is_diff_tournament',
        'home_schedule_next_game_is_high_priority', 'home_schedule_next_opp_elo_vs_league', 'diff_matchday_number'
        # Noise
        'diff_pct_blocks_last_10_blocks_Close_Right', 'diff_pct_blocks_last_5_blocks_Close_Left', 'diff_pct_blocks_last_5_blocks_Close_Right',
        'diff_pct_blocks_last_5_blocks_Mid_Right', 'diff_pct_goals_last_10_goals_Close_Left', 'diff_pct_goals_last_10_goals_Close_Right',
        'diff_pct_goals_last_10_goals_Mid_Left', 'diff_pct_goals_last_5_goals_Close_Left', 'diff_pct_goals_last_5_goals_Close_Right'
        'diff_pct_goals_last_5_goals_Mid_Left', 'diff_pct_goals_last_5_goals_Mid_Right', 'diff_pct_sot_last_10_sot_Close_Right',
        'diff_pct_sot_last_5_sot_Close_Left', 'diff_pct_sot_last_5_sot_Close_Right', 'diff_starters_count', 'fatigue_pace_multiplier',
        'diff_gk_avg_sweeper_actions_last_10', 'diff_gk_avg_sweeper_actions_last_20', 'diff_gk_avg_sweeper_actions_last_5',
        'diff_schedule_is_trap_game', 'diff_roll_5_team_match_top_shooter_shots', 'diff_roll_5_team_match_shot_hhi', 'diff_xi_avg_last_5_shots_taken',
        'diff_n_forwards', 'diff_n_defenders', 'diff_n_midfielders', 'diff_opp_n_midfielders', 'diff_opp_n_forwards', 'diff_opp_n_defenders'
        'home_n_defenders','home_n_forwards','home_n_midfielders','home_opp_n_defenders','home_opp_n_forwards', 'home_opp_n_midfielders',
        'away_n_defenders','away_n_forwards','away_n_midfielders','away_opp_n_defenders','away_opp_n_forwards', 'away_opp_n_midfielders', 'home_n_defenders',
        'home_tactics_changed_formation', 'home_tactics_def_overload', 'home_tactics_instability_index_last_10', 'home_tactics_is_back_three', 'home_tactics_is_lone_striker',
        'home_tactics_midfield_diff', 'venue_id', 'latitude', 'longitude'

        # Cleaning Special Case
        'yellow_cards_elo_post_home', 'yellow_cards_elo_post_away',
        'yellow_cards_elo_delta', 'corners_elo_post_home',
        'corners_elo_post_away', 'corners_elo_delta',
        'total_shots_elo_post_home', 'total_shots_elo_post_away', 'total_shots_elo_delta', 'home_corners', 'home_total_shots', 'away_total_shots', 'away_corners'
    ]


    LEAKY_LIST = [
        'home_score', 'away_score', 'home_goals_scored', 'home_goals_conceded',
        'home_clean_sheet', 'home_expected_goals', 'home_xga', 'home_ppda',
        'home_xg_padj', 'home_buildup_score', 'home_match_points', 'home_is_win',
        'away_goals_scored', 'away_goals_conceded', 'away_clean_sheet',
        'away_expected_goals', 'away_xga', 'away_ppda', 'away_xg_padj',
        'away_buildup_score', 'away_match_points', 'away_is_win',
        'target_total_goals', 'target_btts', 'home_1st_yellow_cards', 'home_2nd_yellow_cards',
        'home_1st_red_cards', 'home_2nd_red_cards', 'home_total_red_cards',
        'home_1st_total_shots', 'home_2nd_total_shots', 'home_1st_shots_on_target',
        'home_2nd_shots_on_target', 'home_total_shots_on_target', 'home_1st_corners', 'home_2nd_corners', 'home_total_corners',
        'away_1st_yellow_cards', 'away_2nd_yellow_cards', 'away_1st_red_cards', 'away_2nd_red_cards', 'away_total_red_cards',
        'away_1st_total_shots', 'away_2nd_total_shots', 'away_1st_shots_on_target', 'away_2nd_shots_on_target',
        'away_total_shots_on_target', 'away_1st_corners', 'away_2nd_corners', 'away_total_corners', 'total_yellow_cards', 'home_corners', 'away_corners',
        'home_shots_faced', 'home_shots_taken', 'away_shots_faced', 'away_shots_taken', 'home_total_shots', 'home_starters_count','diff_xga', 'diff_xg_padj',
        'away_corners', 'home_corners', 'away_total_shots', 'home_total_shots'
    ]

    EXCLUDE_COLS = []
    LEAKY_COLS = []

    for col in EXCLUDE_LIST:
        if col in df.columns:
            EXCLUDE_COLS.append(col)

    for col in LEAKY_LIST:
        if col in df.columns:
            LEAKY_COLS.append(col)

    zone_cols = [col for col in df.columns if 'pct_goals_last_10' in col or 'pct_blocks' in col]

    tactics_cols = [col for col in df.columns if 'tactics' in col]

    penalty_features = [col for col in df.columns if 'pen' in col or 'luck' in col]

    plantilla_features = [col for col in df.columns if 'xi' in col]

    id_cols = [col for col in df.columns if 'id' in col]



    for col in penalty_features:
        if col not in EXCLUDE_COLS:
            EXCLUDE_COLS.append(col)

    for col in plantilla_features:
        if col not in EXCLUDE_COLS:
            EXCLUDE_COLS.append(col)

    for col in zone_cols:
        if col not in EXCLUDE_COLS:
            EXCLUDE_COLS.append(col)

    for col in tactics_cols:
        if col not in EXCLUDE_COLS:
            EXCLUDE_COLS.append(col)


    for col in id_cols:
        if col not in EXCLUDE_COLS:
            EXCLUDE_COLS.append(col)


    for col in LEAKY_LIST:
        if col in df.columns:
            LEAKY_COLS.append(col)

    return EXCLUDE_COLS, LEAKY_COLS, TARGET_COL

