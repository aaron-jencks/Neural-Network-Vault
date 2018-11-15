import pandas as pd
from multiprocessing import Pool

# Class for containing data from the database
class QAObject:
	def __init__(self, questionType: str, asin: str, question: str, answer: str, answerType: str = ""):
		self.QuestionType = questionType
		self.Asin = asin
		self.Question = question
		self.Answer = answer
		self.AnswerType = answerType
		
# Returns a tuple with arrays of questions and answers
def getData(db: list) -> tuple:
	questions = []
	answers = []
	for QA in db:
		questions.append(QA.Question)
		answers.append(QA.Answer)
	return (questions, answers)
	
def get_dataframe(db):
	df = pd.DataFrame(db, columns = ['text', 'label'])
	df.label = df.label.astype('category')
	return df
	