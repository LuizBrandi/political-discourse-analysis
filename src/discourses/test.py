from pathlib import Path
import pandas as pd
from multiprocessing import freeze_support

from ._04_topics import topics_main

def main():
	# antes da eleição
	csv_path_antes = Path(
		"C:/Users/luizf/github/luizbrandi/political-discourse-analysis/data/running_files/political_discourses_ini_02072022_fim_29102022.csv"
	)

	df_csv_antes = pd.read_csv(csv_path_antes)

	topics_main(
		df_csv_antes,
		"UNIÃO",
		topic_limit=10,
		coherence_processes=1,
		search_passes=6,
		search_iterations=60,
		final_passes=20,
		final_iterations=200,
	)


if __name__ == "__main__":
	freeze_support()
	main()