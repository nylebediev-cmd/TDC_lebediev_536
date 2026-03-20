import speech_recognition as srec

def recognize_speech(rec, mic):
    with mic as source:
        print("Скажіть щось...")
        rec.adjust_for_ambient_noise(source, duration=0.5)
        audio = rec.listen(source, timeout=5, phrase_time_limit=5)

    try:
        text = rec.recognize_google(audio, language="uk-UA")
        return text
    except srec.UnknownValueError:
        return "Не вдалося розпізнати мову"
    except srec.RequestError as e:
        return f"Помилка сервісу: {e}"


if __name__ == "__main__":
    recognizer = srec.Recognizer()

    # сначала попробуй БЕЗ device_index
    mic = srec.Microphone()

    while True:
        result = recognize_speech(recognizer, mic)
        print("Ви сказали:", result)