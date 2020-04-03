ISALPHA_VOCABULARY_SIZE = 116766 

def getLocalChar():
    res = ""
    res += "àèéëïĳ"
    res += "áêéèëïíîôóúû"
    res += "æøå"
    res += "åäö"
    res += "äöüß"
    res += "çêîşû"
    res += "ăîâşţ"
    res += "âêîôûŵŷáéíï"
    res += "¡¿áéíñóúü"
    res += "àéèìòù"
    res += "áďéěňóřťúůý"
    res += "áäďéíľĺňóôŕťúý"
    res += "ćęśź"

    return res

def genAlphabetSet(V = 1):
    start = ord('a')
    end = ord('z')
    ans = set()
    for i in range(start, end+1):
        ans.add(chr(i))
    if V == 4:
        for char in getLocalChar():
            ans.add(char)
    if V == 2:
        start = ord('A')
        end = ord('Z')
        for i in range(start, end+1):
            ans.add(chr(i))
    return ans