code:
reviews = [
    "I loved the movie, fantastic!",
    "Worst film ever, boring.",
    "It was okay, not great."
]

actual_sentiments = ["Positive", "Negative", "Neutral"]
predicted_sentiments = ["Positive", "Negative", "Positive"]

print(f'{"Review Text":<35} {"Actual Sentiment":<15} {"Predicted Sentiment":<20} {"Correct (Y/N)"}')
print('-' * 90)

for review, actual, pred in zip(reviews, actual_sentiments, predicted_sentiments):
    correct = 'Y' if actual == pred else 'N'
    print(f'{review:<35} {actual:<15} {pred:<20} {correct}')
    
Output:
<img width="647" height="92" alt="image" src="https://github.com/user-attachments/assets/386ee2ba-2eb1-4494-8628-3e5d1fa07d8e" />
