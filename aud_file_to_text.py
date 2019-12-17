#import speech_recognition as sr
import io
import os
from google.oauth2 import service_account
from google.cloud import speech_v1p1beta1

# time in seconds
time_length_of_audio = 146



def sample_long_running_recognize(storage_uri):
    #variables to count number of words spoken by the each speaker    
    speaker_1_word_cnt = 0
    speaker_2_word_cnt = 0
    
    credentials = service_account.Credentials.from_service_account_file("C://Users//Dinusha//Desktop//fasial Emotion//Speech_to_words//My speech-f0aec7328e19.json")
    client = speech_v1p1beta1.SpeechClient(credentials =credentials)

    # local_file_path = 'resources/commercial_mono.wav'

    # If enabled, each word in the first alternative of each result will be
    # tagged with a speaker tag to identify the speaker.
    enable_speaker_diarization = True

    # Optional. Specifies the estimated number of speakers in the conversation.
    diarization_speaker_count = 2

    # The language of the supplied audio
    language_code = "en-US"
    config = {
        "enable_speaker_diarization": enable_speaker_diarization,
        "diarization_speaker_count": diarization_speaker_count,
        "language_code": language_code,
    }
    audio = {"uri": storage_uri}

    operation = client.long_running_recognize(config, audio)

    print(u"Waiting for operation to complete...")
    response = operation.result()
    
    for result in response.results:
        # First alternative has words tagged with speakers
        alternative = result.alternatives[0]
        print(u"Transcript: {}".format(alternative.transcript))
        # Print the speaker_tag of each word
        for word in alternative.words:
                if word.speaker_tag == 1:
                        speaker_1_word_cnt += 1
                elif word.speaker_tag == 2:
                        speaker_2_word_cnt += 1
                print(u"Word: {}".format(word.word))
                print(u"Speaker tag: {}".format(word.speaker_tag))
                        
    return  speaker_1_word_cnt, speaker_2_word_cnt   

        
storage_uri = "gs://dnb94/audio_file_mono.flac"
#sample_long_running_recognize(storage_uri)
sp1,sp2 = sample_long_running_recognize(storage_uri)

sp1_word_rate = sp1/time_length_of_audio
sp2_word_rate = sp2/time_length_of_audio

print(sp1_word_rate,sp2_word_rate)








