from pathlib import Path
import joblib

from lingua import Language, LanguageDetectorBuilder
from deep_translator import GoogleTranslator


def build_detector():
    languages = [
        Language.ENGLISH,
        Language.CHINESE,
        Language.HINDI,
        Language.SPANISH,
        Language.FRENCH,
        Language.ARABIC,
        Language.BENGALI,
        Language.PORTUGUESE,
        Language.RUSSIAN,
        Language.URDU,
        Language.INDONESIAN,
        Language.GERMAN,
        Language.JAPANESE,
        Language.TURKISH,
        Language.KOREAN,
    ]
    return LanguageDetectorBuilder.from_languages(*languages).build()


def detect_language(detector, text: str) -> str:
    detected = detector.detect_language_of(text)
    if detected is None:
        return "unknown"

    language_map = {
        Language.ENGLISH: "en",
        Language.CHINESE: "zh-CN",
        Language.HINDI: "hi",
        Language.SPANISH: "es",
        Language.FRENCH: "fr",
        Language.ARABIC: "ar",
        Language.BENGALI: "bn",
        Language.PORTUGUESE: "pt",
        Language.RUSSIAN: "ru",
        Language.URDU: "ur",
        Language.INDONESIAN: "id",
        Language.GERMAN: "de",
        Language.JAPANESE: "ja",
        Language.TURKISH: "tr",
        Language.KOREAN: "ko",
    }
    return language_map.get(detected, "unknown")


def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang in ("en", "unknown"):
        return text

    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except Exception:
        return text


def load_model():
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"

    joblib_files = sorted(models_dir.glob("*.joblib"))
    if not joblib_files:
        raise FileNotFoundError("No saved model found in the models/ directory.")

    return joblib.load(joblib_files[0])


def get_ui_text(lang: str) -> dict:
    texts = {
        "en": {
            "detected": "Detected language: English",
            "not_detected": "Could not detect the language. English will be used by default.",
            "warning": "Note: the model was trained on an English dataset, so accuracy may be lower for other languages.",
            "prompt": "Enter the message you want to check:",
            "menu": "Type your message, or enter '/lang' to detect a new language, or '/exit' to quit.",
            "prediction_spam": "Prediction: likely spam",
            "prediction_ham": "Prediction: likely not spam",
            "translated": "Translated to English:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "Please enter any text so I can detect your language:",
            "language_reset": "Language detection restarted.",
            "goodbye": "Goodbye!",
        },
        "ru": {
            "detected": "Определённый язык: русский",
            "not_detected": "Не удалось определить язык. По умолчанию будет использоваться английский.",
            "warning": "Обратите внимание: модель обучалась на английском датасете, поэтому для других языков точность может быть ниже.",
            "prompt": "Введите сообщение, которое нужно проверить:",
            "menu": "Введите сообщение или команду '/lang' для повторного определения языка, или '/exit' для выхода.",
            "prediction_spam": "Результат: похоже на спам",
            "prediction_ham": "Результат: похоже не на спам",
            "translated": "Перевод на английский:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "Введите любой текст, чтобы я определил язык:",
            "language_reset": "Определение языка запущено заново.",
            "goodbye": "До свидания!",
        },
        "es": {
            "detected": "Idioma detectado: español",
            "not_detected": "No se pudo detectar el idioma. Se usará inglés por defecto.",
            "warning": "Nota: el modelo fue entrenado con un conjunto de datos en inglés, por lo que la precisión puede ser menor para otros idiomas.",
            "prompt": "Introduce el mensaje que quieres comprobar:",
            "menu": "Escribe un mensaje, o '/lang' para detectar otro idioma, o '/exit' para salir.",
            "prediction_spam": "Resultado: probablemente spam",
            "prediction_ham": "Resultado: probablemente no es spam",
            "translated": "Traducido al inglés:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "Escribe cualquier texto para que pueda detectar tu idioma:",
            "language_reset": "La detección de idioma se ha reiniciado.",
            "goodbye": "¡Adiós!",
        },
        "fr": {
            "detected": "Langue détectée : français",
            "not_detected": "Impossible de détecter la langue. L'anglais sera utilisé par défaut.",
            "warning": "Remarque : le modèle a été entraîné sur un jeu de données en anglais, donc la précision peut être plus faible pour les autres langues.",
            "prompt": "Entrez le message à vérifier :",
            "menu": "Entrez un message, ou '/lang' pour redétecter la langue, ou '/exit' pour quitter.",
            "prediction_spam": "Résultat : probablement du spam",
            "prediction_ham": "Résultat : probablement pas du spam",
            "translated": "Traduit en anglais :",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "Veuillez saisir un texte pour que je détecte votre langue :",
            "language_reset": "La détection de la langue a été relancée.",
            "goodbye": "Au revoir !",
        },
        "de": {
            "detected": "Erkannte Sprache: Deutsch",
            "not_detected": "Die Sprache konnte nicht erkannt werden. Standardmäßig wird Englisch verwendet.",
            "warning": "Hinweis: Das Modell wurde mit einem englischen Datensatz trainiert, daher kann die Genauigkeit für andere Sprachen geringer sein.",
            "prompt": "Geben Sie die Nachricht ein, die überprüft werden soll:",
            "menu": "Geben Sie eine Nachricht ein oder '/lang' für eine neue Spracherkennung oder '/exit' zum Beenden.",
            "prediction_spam": "Ergebnis: wahrscheinlich Spam",
            "prediction_ham": "Ergebnis: wahrscheinlich kein Spam",
            "translated": "Ins Englische übersetzt:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "Bitte geben Sie einen beliebigen Text ein, damit ich Ihre Sprache erkennen kann:",
            "language_reset": "Spracherkennung wurde neu gestartet.",
            "goodbye": "Auf Wiedersehen!",
        },
        "pt": {
            "detected": "Idioma detectado: português",
            "not_detected": "Não foi possível detectar o idioma. O inglês será usado por padrão.",
            "warning": "Observação: o modelo foi treinado com um conjunto de dados em inglês, então a precisão pode ser menor para outros idiomas.",
            "prompt": "Digite a mensagem que deseja verificar:",
            "menu": "Digite uma mensagem, ou '/lang' para detectar outro idioma, ou '/exit' para sair.",
            "prediction_spam": "Resultado: provavelmente spam",
            "prediction_ham": "Resultado: provavelmente não é spam",
            "translated": "Traduzido para o inglês:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "Digite qualquer texto para que eu possa detectar seu idioma:",
            "language_reset": "A detecção de idioma foi reiniciada.",
            "goodbye": "Tchau!",
        },
        "zh-CN": {
            "detected": "检测到的语言：中文",
            "not_detected": "无法检测语言。将默认使用英语。",
            "warning": "注意：该模型是在英文数据集上训练的，因此对其他语言的准确率可能较低。",
            "prompt": "请输入要检查的消息：",
            "menu": "请输入消息，或输入 '/lang' 重新检测语言，或输入 '/exit' 退出。",
            "prediction_spam": "结果：可能是垃圾信息",
            "prediction_ham": "结果：可能不是垃圾信息",
            "translated": "翻译成英文：",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "请输入任意文本，以便我检测你的语言：",
            "language_reset": "语言检测已重新启动。",
            "goodbye": "再见！",
        },
        "hi": {
            "detected": "पहचानी गई भाषा: हिन्दी",
            "not_detected": "भाषा पहचान में नहीं आई। डिफ़ॉल्ट रूप से अंग्रेज़ी का उपयोग किया जाएगा।",
            "warning": "ध्यान दें: मॉडल को अंग्रेज़ी डेटासेट पर प्रशिक्षित किया गया था, इसलिए अन्य भाषाओं के लिए सटीकता कम हो सकती है।",
            "prompt": "जाँचने के लिए संदेश दर्ज करें:",
            "menu": "संदेश लिखें, या भाषा फिर से पहचानने के लिए '/lang', या बाहर निकलने के लिए '/exit' लिखें।",
            "prediction_spam": "परिणाम: संभवतः स्पैम",
            "prediction_ham": "परिणाम: संभवतः स्पैम नहीं",
            "translated": "अंग्रेज़ी में अनुवाद:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "कृपया कोई भी पाठ दर्ज करें ताकि मैं आपकी भाषा पहचान सकूँ:",
            "language_reset": "भाषा पहचान फिर से शुरू की गई।",
            "goodbye": "अलविदा!",
        },
        "ar": {
            "detected": "اللغة المكتشفة: العربية",
            "not_detected": "تعذر اكتشاف اللغة. سيتم استخدام الإنجليزية افتراضيًا.",
            "warning": "ملاحظة: تم تدريب النموذج على مجموعة بيانات باللغة الإنجليزية، لذلك قد تكون الدقة أقل للغات الأخرى.",
            "prompt": "أدخل الرسالة التي تريد التحقق منها:",
            "menu": "اكتب رسالة، أو أدخل '/lang' لإعادة اكتشاف اللغة، أو '/exit' للخروج.",
            "prediction_spam": "النتيجة: على الأرجح رسالة مزعجة",
            "prediction_ham": "النتيجة: على الأرجح ليست رسالة مزعجة",
            "translated": "الترجمة إلى الإنجليزية:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "يرجى إدخال أي نص حتى أتمكن من اكتشاف لغتك:",
            "language_reset": "تمت إعادة تشغيل اكتشاف اللغة.",
            "goodbye": "مع السلامة!",
        },
        "bn": {
            "detected": "সনাক্ত ভাষা: বাংলা",
            "not_detected": "ভাষা সনাক্ত করা যায়নি। ডিফল্ট হিসেবে ইংরেজি ব্যবহার করা হবে।",
            "warning": "দ্রষ্টব্য: মডেলটি ইংরেজি ডেটাসেটে প্রশিক্ষিত, তাই অন্য ভাষার ক্ষেত্রে সঠিকতা কম হতে পারে।",
            "prompt": "যে বার্তাটি পরীক্ষা করতে চান তা লিখুন:",
            "menu": "বার্তা লিখুন, অথবা ভাষা আবার শনাক্ত করতে '/lang', অথবা বের হতে '/exit' লিখুন।",
            "prediction_spam": "ফলাফল: সম্ভবত স্প্যাম",
            "prediction_ham": "ফলাফল: সম্ভবত স্প্যাম নয়",
            "translated": "ইংরেজিতে অনুবাদ:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "অনুগ্রহ করে যেকোনো লেখা লিখুন, যাতে আমি আপনার ভাষা শনাক্ত করতে পারি:",
            "language_reset": "ভাষা শনাক্তকরণ আবার শুরু হয়েছে।",
            "goodbye": "বিদায়!",
        },
        "ur": {
            "detected": "شناختہ زبان: اردو",
            "not_detected": "زبان معلوم نہیں ہو سکی۔ بطور ڈیفالٹ انگریزی استعمال ہوگی۔",
            "warning": "نوٹ: ماڈل کو انگریزی ڈیٹاسیٹ پر تربیت دی گئی تھی، اس لیے دوسری زبانوں کے لیے درستگی کم ہو سکتی ہے۔",
            "prompt": "وہ پیغام درج کریں جسے آپ چیک کرنا چاہتے ہیں:",
            "menu": "پیغام درج کریں، یا زبان دوبارہ معلوم کرنے کے لیے '/lang'، یا باہر نکلنے کے لیے '/exit' درج کریں۔",
            "prediction_spam": "نتیجہ: غالباً اسپام",
            "prediction_ham": "نتیجہ: غالباً اسپام نہیں",
            "translated": "انگریزی میں ترجمہ:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "براہ کرم کوئی بھی متن درج کریں تاکہ میں آپ کی زبان معلوم کر سکوں:",
            "language_reset": "زبان کی شناخت دوبارہ شروع کر دی گئی ہے۔",
            "goodbye": "خدا حافظ!",
        },
        "id": {
            "detected": "Bahasa terdeteksi: Indonesia",
            "not_detected": "Bahasa tidak dapat dideteksi. Bahasa Inggris akan digunakan secara default.",
            "warning": "Catatan: model dilatih pada dataset berbahasa Inggris, jadi akurasi bisa lebih rendah untuk bahasa lain.",
            "prompt": "Masukkan pesan yang ingin diperiksa:",
            "menu": "Ketik pesan, atau masukkan '/lang' untuk mendeteksi ulang bahasa, atau '/exit' untuk keluar.",
            "prediction_spam": "Hasil: kemungkinan spam",
            "prediction_ham": "Hasil: kemungkinan bukan spam",
            "translated": "Diterjemahkan ke bahasa Inggris:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "Silakan masukkan teks apa pun agar saya dapat mendeteksi bahasa Anda:",
            "language_reset": "Deteksi bahasa dimulai ulang.",
            "goodbye": "Sampai jumpa!",
        },
        "ja": {
            "detected": "検出された言語: 日本語",
            "not_detected": "言語を検出できませんでした。デフォルトで英語を使用します。",
            "warning": "注意: このモデルは英語のデータセットで学習されているため、他の言語では精度が下がる可能性があります。",
            "prompt": "確認したいメッセージを入力してください:",
            "menu": "メッセージを入力するか、言語を再検出するには '/lang'、終了するには '/exit' を入力してください。",
            "prediction_spam": "結果: スパムの可能性があります",
            "prediction_ham": "結果: スパムではない可能性があります",
            "translated": "英語に翻訳:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "言語を検出するために任意のテキストを入力してください:",
            "language_reset": "言語検出を再開しました。",
            "goodbye": "さようなら！",
        },
        "tr": {
            "detected": "Algılanan dil: Türkçe",
            "not_detected": "Dil tespit edilemedi. Varsayılan olarak İngilizce kullanılacak.",
            "warning": "Not: model İngilizce bir veri kümesi üzerinde eğitildi, bu nedenle diğer diller için doğruluk daha düşük olabilir.",
            "prompt": "Kontrol etmek istediğiniz mesajı girin:",
            "menu": "Bir mesaj yazın, veya dili yeniden tespit etmek için '/lang', çıkmak için '/exit' girin.",
            "prediction_spam": "Sonuç: muhtemelen spam",
            "prediction_ham": "Sonuç: muhtemelen spam değil",
            "translated": "İngilizceye çevrildi:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "Dilini tespit edebilmem için lütfen herhangi bir metin gir:",
            "language_reset": "Dil algılama yeniden başlatıldı.",
            "goodbye": "Hoşça kal!",
        },
        "ko": {
            "detected": "감지된 언어: 한국어",
            "not_detected": "언어를 감지할 수 없습니다. 기본적으로 영어가 사용됩니다.",
            "warning": "참고: 이 모델은 영어 데이터셋으로 학습되었으므로 다른 언어에서는 정확도가 더 낮을 수 있습니다.",
            "prompt": "확인할 메시지를 입력하세요:",
            "menu": "메시지를 입력하거나, 언어를 다시 감지하려면 '/lang', 종료하려면 '/exit'를 입력하세요.",
            "prediction_spam": "결과: 스팸일 가능성이 높습니다",
            "prediction_ham": "결과: 스팸이 아닐 가능성이 높습니다",
            "translated": "영어로 번역됨:",
            "hello": "Hello! I can help you determine whether a message is likely to be spam or not. Please enter any message so that I can detect your language (please note that the model was trained on an English dataset, so accuracy may be lower for other languages).",
            "choose_again": "언어를 감지할 수 있도록 아무 텍스트나 입력하세요:",
            "language_reset": "언어 감지가 다시 시작되었습니다.",
            "goodbye": "안녕히 가세요!",
        },
    }

    return texts.get(lang, texts["en"])


def classify_message(model, text: str, lang: str, ui: dict) -> None:
    translated_text = translate_to_english(text, lang)

    if lang != "en":
        print(f"{ui['translated']} {translated_text}")

    prediction = model.predict([translated_text])[0]

    if prediction == "spam":
        print(ui["prediction_spam"])
    else:
        print(ui["prediction_ham"])


def language_session(model, lang: str) -> str:
    ui = get_ui_text(lang)

    if lang == "unknown":
        ui = get_ui_text("en")
        print(ui["not_detected"])
        lang = "en"
        ui = get_ui_text(lang)
    else:
        print(ui["detected"])

    print(ui["warning"])
    print(ui["menu"])

    while True:
        text = input(f"\n{ui['prompt']} ").strip()

        if not text:
            continue

        if text.lower() == "/exit":
            print(ui["goodbye"])
            return "exit"

        if text.lower() == "/lang":
            print(ui["language_reset"])
            return "reset"

        classify_message(model, text, lang, ui)


def main():
    model = load_model()
    detector = build_detector()

    print(get_ui_text("en")["hello"])

    while True:
        seed_text = input("\n" + get_ui_text("en")["choose_again"] + " ").strip()

        if not seed_text:
            continue

        if seed_text.lower() == "/exit":
            print(get_ui_text("en")["goodbye"])
            break

        detected_lang = detect_language(detector, seed_text)

        session_result = language_session(model, detected_lang)

        if session_result == "exit":
            break


if __name__ == "__main__":
    main()