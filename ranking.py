#basic Program needs


# Calculate the minimum edit distance
def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[
                             j + 1] + 1  # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1  # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

#Getting similarity co-efficient between two strings
def GetSimilarity(string1,string2):
    dis=levenshtein(string1,string2)
    maxLen =len(string1)
    if (maxLen < len(string2)):
        maxLen=len(string2)
        if (maxLen == 0):
            return 1
        else:
            return 1 - dis/maxLen;
    else:
        if (maxLen == 0):
            return 1
        else:
            return 1 - dis/maxLen;



#Main Program starts from here

string1= "string1"
string2= "string2"

final=GetSimilarity(string1,string2)
print(final)
