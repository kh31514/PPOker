import pandas as pd
import os


def save_to_csv(data, folder_name):
    path = f"./results/{folder_name}"
    os.makedirs(path, exist_ok=True)

    win_reward_df = pd.DataFrame()
    win_reward_df["steps"] = data["steps"]

    for opp in data:
        if opp != "steps":
            win_reward_df[opp + "_" + "win_rate"] = data[opp]["win_rate"]
            win_reward_df[opp + "_" +
                          "total_reward"] = data[opp]["total_reward"]

            move_df = pd.DataFrame(data[opp]["move_rate"], columns=[
                'fold', 'check/all', 'raise half', 'raise full', 'all in'])
            move_df["steps"] = data["steps"]
            move_df.to_csv(path + "/" + opp + "_" + "move_count.csv",
                           index=False, header=True)

    win_reward_df.to_csv(path + "/" + "win_reward.csv",
                         index=False, header=True)

    return
