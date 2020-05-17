import Levenshtein


def edit_distance_str(str1, str2):
    edit_distance_distance = Levenshtein.distance(str1, str2)
    similarity = 1 - (edit_distance_distance / max(len(str1), len(str2)))
    return {'Distance': edit_distance_distance, 'Similarity': similarity}
