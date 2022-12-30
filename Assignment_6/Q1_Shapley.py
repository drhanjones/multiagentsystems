import random
import copy
import pandas as pd


def shapley_theoretical_value(n, participant_num):
    shap_val = 0
    for i in range(participant_num):
        #print(i, participant_num, n, f"1/{n}-{i}")
        shap_val += 1/(n-i)
    #print(shap_val)

    return shap_val


def shapley_sample(participant, participant_list, distance_dict):

    drop_order = copy.deepcopy(participant_list)
    random.shuffle(drop_order)
    participant_index = drop_order.index(participant)
    prev_drop = drop_order[:participant_index]
    p_val = distance_dict[participant]
    #print("H3",participant,drop_order, participant_index,prev_drop, p_val, "H4")
    if len(prev_drop) == 0:
        return distance_dict[participant]
    max_distance = max([distance_dict[drop] for drop in prev_drop])

    if max_distance > p_val:
        shapley_value = 0
    else:
        shapley_value = p_val - max_distance

    return shapley_value

def shapley_monte_carlo(participant, participant_list, distance_dict,  num_samples):


    sample_list = [shapley_sample(participant, participant_list, distance_dict) for _ in range(num_samples)]
    #print("s1", sample_list)

    shapley_value = sum(sample_list)/num_samples

    return shapley_value

def shapley_exact(n):
    participants_distance = {"n" + str(i): i for i in range(1, n + 1)}
    participant_list = list(participants_distance.keys())

    shapley_dict = {}
    for participant in participant_list:
        #print(participant)
        shapley_dict[participant] = shapley_theoretical_value(n, participants_distance[participant])

    return shapley_dict

def shapley_mc(n, num_samples):
    participants_distance = {"n" + str(i): i for i in range(1, n + 1)}
    participant_list = list(participants_distance.keys())
    #print("h1", participant_list)
    shapley_dict = {}
    for participant in participant_list:
        #print(participant)
        shapley_dict[participant] = shapley_monte_carlo(participant, participant_list, participants_distance, num_samples)

    return shapley_dict

def main():
    n = 50
    s_exact = shapley_exact(n)
    print(s_exact)

    num_samples = 1000
    s_mc = shapley_mc(n, num_samples)
    print(s_mc)
    df = pd.DataFrame({"exact": s_exact, "monte carlo": s_mc})
    df["difference"] = df["exact"] - df["monte carlo"]
    print(df)
    print(df.sum(axis=0))


if __name__ == "__main__":
    main()
