# -*- coding: utf-8 -*-
"""
environ : nlpr


"""

import pandas as pd
import numpy as np
import spacy

from spacy.matcher import PhraseMatcher

import re


from dateparser.search import search_dates

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

from collections import Counter
from tika import parser

#cvname='cva.docx'


import os
#pool = mp.Pool(mp.cpu_count())

Candidates_list={}
resumes = []
data = []
for root, directories, filenames in os.walk('.'):  #linux
    for filename in filenames:
        file = os.path.join(root, filename)
        if file.lower().endswith(('.pdf', '.docx', '.doc')):
            resumes.append(file)
        
        
summary2="allfiles.sum"    
ff = open(summary2,'w+',encoding="utf-8") 
    
           
#resumes=['c:\\resume1\\cv7.pdf']       

#for cvname in [ 'cv7.pdf', 'cv6.pdf', 'cv5.pdf', 'cv4.pdf', 'cv3.pdf', 'cv2.pdf', 'cv1.pdf', 'cva.docx',  'cvb.docx',  'cvc.docx'] :    
for cvname in resumes:    
    parsed = parser.from_file(cvname)
#    print(parsed["metadata"])
#    print(parsed["content"])
    print(cvname)
    
     
    
    print(cvname, file=ff)  
    cv1=parsed["content"]
    cv1m=parsed["metadata"]
    
    def cleanResume(resumeText):
        resumeText1=resumeText.strip()
        resumeText2 = re.sub(' +', ' ', resumeText1)  # remove extra whitespace
        
        line_list = resumeText2.split("\n")
        number_of_lines = len(line_list)
#        print("lines in cv",number_of_lines) 
        cleaned=[]
        for line in line_list:
            
#            line1= re.split('[;,\s]+', line)
#            line1=re.sub("[$@&●]","",line)
            line1=re.sub("[●]","",line)
            line2=line1.lstrip()
           
            line=line2
            if len(line)>=4:
                if ' page ' not in line.lower(): # remove page'
             
                    cleaned.append(line)
            
        resumeText2='\n'.join(cleaned)
        
#        return resumeText2, number_of_lines,cleaned

#        check if 'page' appears


#second clean remove 'page'

        return cleaned, number_of_lines,cleaned       
    
    cvtext, zlines,line_list =  cleanResume(cv1)  #cvtext lines with text at least 4

    countl=Counter(line_list) #check if lines repeated
    set_lines=set(line_list)
    
    lines_repeated=len(line_list)-len(set_lines) #DEFIN
    lines_repeated1=15*lines_repeated/(len(set_lines)+1) #DEFIN
    
    
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')    
    print("lines in cv",len(line_list)) 
    print('lines reapeated', len(line_list)-len(set_lines))
    
    print("lines in cv",len(line_list),file=ff) 
    print('lines reapeated', len(line_list)-len(set_lines),file=ff)
     
    
    
   #extract_languages(cvtext)    
    
    # load pre-trained model for all the functions.
    nlp = spacy.load('en_core_web_sm')
       
    
    ## CSV academic titles to list
    academic_titles = pd.read_csv('academic_titles.txt', encoding = 'utf-8', header = None)
    academic_titles[0] = academic_titles[0].str.lower()
    academic_titles_words = academic_titles[0].tolist()
    academic_titles = [nlp.make_doc(text) for text in academic_titles_words]
    
    ## CSV academic titles to list
    academic_institutions = pd.read_csv('academic_institutions1.txt', encoding = 'utf-8', header = None)
    academic_institutions[0] = academic_institutions[0].str.lower()
    academic_institutions_words = academic_institutions[0].tolist()    
    academic_institutions = [nlp.make_doc(text) for text in academic_institutions_words]
    
    ## CSV academic titles to list
    qa_skills = pd.read_csv('qa.txt', encoding = 'utf-8', header = None)
    qa_skills[0] = qa_skills[0].str.lower()
    qa_skills_words = qa_skills[0].tolist()    
    qa_skills = [nlp.make_doc(text) for text in qa_skills_words]
    
        
    
    
    
    ##  CSV job titles to list
    job_titles = pd.read_csv('jobs_titles.csv', encoding = 'utf-8', header = None)
    job_titles[0] = job_titles[0].str.lower()
    job_titles_words = job_titles[0].tolist()
    job_titles = [nlp.make_doc(text) for text in job_titles_words]
    
    ##  CSV hard skills to list
    certifications = pd.read_csv('skills0.txt', sep=';', encoding = 'utf-8', header = None)
    certifications[0] = certifications[0].str.lower()
    certifications_words = certifications[0].tolist()
    certifications = [nlp.make_doc(text) for text in certifications_words]
    
    ## CSV hobbies to list
    hobbies = pd.read_csv('hobbies.csv', encoding = 'utf-8', header = None)
    hobbies[0] = hobbies[0].str.lower()
    hobbies_words = hobbies[0].tolist()
    hobbies = [nlp.make_doc(text) for text in hobbies_words]
    
    ## CSV hobbies to list
    languages = pd.read_csv('languages.csv', encoding = 'utf-8', header = None)
    languages[0] = languages[0].str.lower()
    languages_words = languages[0].tolist()
    languages = [nlp.make_doc(text) for text in languages_words]
    
    #Function to extract name
    def name_via_entity(var):
        doc = nlp(var)
        result = []
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                result.append(ent.text)
            else:
                next
        if len(result) > 0:
            return str(result[0])
        else:
            return 'no name'
    
    #Function to extract email
    def get_email_addresses(var):
        r = re.compile(r'[\w\.-]+@[\w\.-]+')
        res = r.findall(str(var))
        if len(res) > 0:
            return res[0]
        else:
            return 'no email'
        
    #Function to extract phonenumber
    def get_phone(var):
        r = re.compile(r"(\+48)?\s*?(\d{9})")
        res = r.findall(str(var))
        if len(res) > 0:
            out = res[0]
            out = '-'.join(out)
        else:
            out = 'no phone'
        return out
    
    #Function to extract the city from address
    def extract_location(col_address):
        doc = nlp(col_address)
        result = []
        for ent in doc.ents:
            if ent.label_=='GPE':
                result.append(ent.text)
            else:
                next
        if len(result) > 0:
            return str(result[0])
        else:
            return 'no address'
    
    ## Function to extract academic titles
    def extract_academic_titles(col_address):
        doc = nlp(col_address)
        phrase_matcher_academic_titles = PhraseMatcher(nlp.vocab)
        phrase_matcher_academic_titles.add("titles", None, *academic_titles)
        matches_academic_titles = phrase_matcher_academic_titles(doc)
        
        academic_list = []
        for match_id, start, end in matches_academic_titles:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            academic_list.append(span.text)
        if len(academic_list) > 0:
            return academic_list,start,end
        else:
            return ['no degree'],0,0
    
    def extract_academic_institutions(col_address):
        doc = nlp(col_address)
        phrase_matcher_academic_institutions = PhraseMatcher(nlp.vocab)
        phrase_matcher_academic_institutions.add("inst", None, *academic_institutions)
        matches_academic_institutions = phrase_matcher_academic_institutions(doc)
        
        academic_institutionsl = []
        for match_id, start, end in matches_academic_institutions:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            academic_institutionsl.append(span.text)
        if len(academic_institutionsl) > 0:
            return academic_institutionsl, start,end
        else:
            return ['no academic institutions'],0,0
        
    ## Function to extract qa skills        
    def extract_qa_skills(col_address):
        doc = nlp(col_address)
        phrase_matcher_qa = PhraseMatcher(nlp.vocab)
        phrase_matcher_qa.add("qa", None, *qa_skills)
        matches_qa = phrase_matcher_qa(doc)
        
        qal = []
        for match_id, start, end in matches_qa:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            qal.append(span.text)
        if len(qal) > 0:
            return qal, start,end
        else:
            return ['no qa skills'],0,0
           
     
    ## Function to extract work experience
    def extract_work_experience(col_address):
        doc = nlp(col_address)
        phrase_matcher_job_titles = PhraseMatcher(nlp.vocab)
        phrase_matcher_job_titles.add("work", None, *job_titles)
        matches_job_titles = phrase_matcher_job_titles(doc)
        
        job_titles_list = []
        for match_id, start, end in matches_job_titles:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            job_titles_list.append(span.text)
        if len(job_titles_list) > 0:
            return job_titles_list
        else:
            return ['no work experience']
    
    def extract_certifications(col_address):
        doc = nlp(col_address)
        phrase_matcher_certifications = PhraseMatcher(nlp.vocab)
        phrase_matcher_certifications.add("certif", None, *certifications)
        matches_certifications = phrase_matcher_certifications(doc)
        
        certifications_list = []
        for match_id, start, end in matches_certifications:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            certifications_list.append(span.text)
        if len(certifications_list) > 0:
            return certifications_list  # get lines also
        else:
            return ['no certifications']
    
    ## Function to extract hobbies
    def extract_hobbies(col_address):
        doc = nlp(col_address)
        phrase_matcher_hobbies = PhraseMatcher(nlp.vocab)
        phrase_matcher_hobbies.add("Hobbies", None, *hobbies)
        matches_hobbies = phrase_matcher_hobbies(doc)
        
        hobbies_list = []
        for match_id, start, end in matches_hobbies:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            hobbies_list.append(span.text)
        if len(hobbies_list) > 0:
            return hobbies_list
        else:
            return [' no hobbies']
        
    ## Function to extract languages
    def extract_languages(col_address):
        doc = nlp(col_address)
        phrase_matcher_languages = PhraseMatcher(nlp.vocab)
        phrase_matcher_languages.add("languages", None, *languages)
        matches_languages = phrase_matcher_languages(doc)
        
        languages_list = []
        for match_id, start, end in matches_languages:
            string_id = nlp.vocab.strings[match_id]
            span = doc[start:end]
            languages_list.append(span.text)
        if len(languages_list) > 0:
            return languages_list
        else:
            return ['Polish']
        
        
        
    cvtextl = [x.lower() for x in cvtext]    #remove same lines
#    cvtextl=cvtext.lower()  

# try simple matching for fields 
    nbr_lines=len(cvtextl)
    
    edu_line=nbr_lines

#    for i in range(nbr_lines) :
#        zz=cvtextl[i].lstrip()
#        cvtextl[i]=zz
#        cvtextl[i].lstrip().strip()

    for i in range(nbr_lines) :
        if cvtextl[i].startswith('education'):
            edu_line=i
            break
 
    print("edu_line", edu_line)





    cvtextl1='\n'.join(cvtextl)    
    doc = nlp(cvtextl1)
    
    nbr_tokens=len(doc)
    
    fields = [ 'education', 'work experience', 'employment', 'professional experience', 'skills' ]
     
    cvfields=[nlp.make_doc(text) for text in fields]            

    education_field = [ 'education', 'studies' ]

    print('nbr of tokens',nbr_tokens)    
    print("nbr of tokens",nbr_tokens, file=ff)           
    
    for token in doc:
        # actual word
        word = token.text
        # lemma
        # part of speech
        word_pos = token.pos_
        if word in fields:
            print(word, word_pos, token.i)    # which tokens
            
            
    for token in doc:
        # actual word
        word = token.text
        # lemma
        # part of speech
        word_pos = token.pos_
        if word in education_field: #check if start of line!!!
            print(word, word_pos, token.i)    # which tokens        
    
    
# check if these fields appear on cv 
    for s in fields:
        res = re.search(s,doc.text)
        if res:
            print(s ,'characters', res.start(), res.end())   # which characters
   
   
    
    zlang=extract_languages(cvtextl1)
    
    nber_lang=len(zlang) #DEFIN
    zhobb=extract_hobbies(cvtextl1)
# search after 'eduction' if found    
    ztitle,ztitles, ztitlee=extract_academic_titles(cvtextl1)
    
    zexper=extract_work_experience(cvtextl1)
    zexperience1=sorted(set(zexper))
    nber_experience=len(zexperience1)  #DEFI
    
    zcert=extract_certifications(cvtextl1)
    zcertifi1=sorted(set(zcert))
    nber_zcertifi1=len(zcertifi1)  #DEFI    
    
    zschool,zshools, zschoole=extract_academic_institutions(cvtextl1)
    
    zqaskills=extract_qa_skills(cvtextl1)
    
    #education data
    
    zzz=nlp(cvtextl1)
    no_educ=0
    if (ztitles * zshools >1):
        no_educ=0
        educ_st=min([ztitles, zshools])
        educ_en=max([ztitlee, zschoole])
        educ_cont=zzz[educ_st-20: educ_en+10].text #use for education context
    
    if (ztitles + zshools <1):
        no_educ=1
         
    if (ztitles +zshools >1):
        no_educ=0
        educ_st=max([ztitles, zshools])
        educ_en=max([ztitlee, zschoole])
        educ_cont=zzz[educ_st-20: educ_en+10].text #use for education context    
        
    
    
    var1=''.join(line_list[0:10])
    
    zname= name_via_entity(var1)
     
#check for dates 19... 20...    
    i=0
    j=0
    for line in cvtext:
        

        if (re.search('[1-2]\d{3}', line)):              # dates
                print('line' ,i)
                print(search_dates(line))
                j+=1
        i+=1
        
        
    nbr_dates=j
    nber_dates=j  #DEFI        
    
    print( "found dates",j)
    print("found dates",j, file=ff)          
    
    modelname = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    model = AutoModelForQuestionAnswering.from_pretrained(modelname)
    
    model.base_model.config
    
    
    
    
    def get_top_answers(possible_starts,possible_ends,input_ids):
      answers = []
      for start,end in zip(possible_starts,possible_ends):
        
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start:end+1]))
        answers.append( answer )
      return answers  
    
    def answer_question(question,context,topN):
    
        inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
        
        input_ids = inputs["input_ids"].tolist()[0]
    
        text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        model_out = model(**inputs)
         
        answer_start_scores = model_out["start_logits"]
        answer_end_scores = model_out["end_logits"]
    
        possible_starts = np.argsort(answer_start_scores.cpu().detach().numpy()).flatten()[::-1][:topN]
        possible_ends = np.argsort(answer_end_scores.cpu().detach().numpy()).flatten()[::-1][:topN]
        
        #get best answer
        answer_start = torch.argmax(answer_start_scores)  
        answer_end = torch.argmax(answer_end_scores) + 1  # Get the most likely end of answer with the argmax of the score
    
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    
        answers = get_top_answers(possible_starts,possible_ends,input_ids )
    
        return { "answer":answer,"answer_start":answer_start,"answer_end":answer_end,"input_ids":input_ids,
                "answer_start_scores":answer_start_scores,"answer_end_scores":answer_end_scores,"inputs":inputs,"answers":answers,
                "possible_starts":possible_starts,"possible_ends":possible_ends}
    
    questions_edu = [
        " Does he have a degree ? ",
        "What did he study ?",
        "When did he finish his education ?",
        ]
    
    
    questions_work = [
    " Did he work with testing ? ",
    "Did he work with java ?",
    " Did he do Quality Assurance ?",
    
    ]
        
    
    summary=cvname+".sum"

    f = open(summary,'w+',encoding="utf-8")
    
 
    
#    print(cvname, file=ff)       
    
    print(len(zcertifi1),' certificates \n', file=f)
    print(len(zcertifi1),' certificates \n', file=ff)    
    print(zcertifi1, file=f)
    print(zcertifi1, file=ff)    
    
    
    print(len(zexperience1), ' experience \n', file=f)
    print(zexperience1, file=f)
    
    print(len(zexperience1), ' experience \n', file=ff)
    print(zexperience1, file=ff)    
    
    
    
    
    print('\n languages \n', file=f)
    print(set(zlang), file=f)
    print('\n schools \n', file=f)
    print(zschool, file=f)
    
    print('\n languages \n', file=ff)
    print(set(zlang), file=ff)
    print('\n schools \n', file=ff)
    print(zschool, file=ff)    
    
    
    
    
    
    
    print('\n degree \n', file=f)
    print(ztitle, file=f)
    print('\n degree \n', file=ff)
    print(ztitle, file=ff)    
    print('\n qa skills \n', file=f)
    print(zqaskills, file=f)    
    print('\n qa skills \n', file=ff)
    print(zqaskills, file=ff)        
    
    nber_degree=3*len(ztitle)+5
    if ztitle[0] =='no degree' :
        nber_degree=0
    
    
    if edu_line<nbr_lines:    # education found!!! search for max 8 lines
        search=min(nbr_lines-edu_line, 8)
        educ_cont='\n'.join(cvtextl[edu_line:edu_line+search])
        no_educ=0
    
    
    if no_educ==0:
         
    #    var2=''.join(line_list[95:105])  # find where education is -approximately
        var2=educ_cont #context
        i=0;
        for q in questions_edu:
          answer_map = answer_question(q,var2,5)   
          i+=1
          print( "\n", i, ".Question:",q,file=f)
          print( "\n", i, ".Question:",q,file=ff)         
          print("Answers:", file=f)
          print("Answers:", file=ff)
        #  [print((index+1)," ) ",ans) for index,ans in  enumerate(answer_map["answers"]) if len(ans) > 0 ]
          [print(ans, file=f) for index,ans in  enumerate(answer_map["answers"]) if len(ans) > 0 ]
          [print(ans, file=ff) for index,ans in  enumerate(answer_map["answers"]) if len(ans) > 0 ]
        
    
    points_qa=len(zqaskills[0])    
    nber_qaskills=points_qa #DEFI
    
    nber_mesure1=nber_qaskills -int(lines_repeated1) + 2*nber_lang + int(nber_dates/2) +int(nber_experience/10) +  int(nber_zcertifi1/20)                              #DEFI
    
    candidate_name=cvname
    candidate_features=(nber_mesure1,points_qa,ztitle, zqaskills[0],j)
    new={candidate_name:candidate_features}
    Candidates_list.update(new)    
    f.close()


sort_Candidates = sorted(Candidates_list.items(), key=lambda x: x[1], reverse=True)


for measure in sort_Candidates:
     print (measure[0],', my score=',measure[1][0],', qa score=', measure[1][1],measure[1][2])
     print (measure[0],', my score=',measure[1][0],', qa score=', measure[1][1],measure[1][2],file=ff)
     
     
ff.close()     