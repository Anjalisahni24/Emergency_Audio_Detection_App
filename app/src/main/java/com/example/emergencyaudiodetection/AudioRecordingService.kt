package com.example.emergencyaudiodetection

import android.Manifest
import android.content.pm.PackageManager
import android.app.Service
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.IBinder
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import android.util.Log
import android.content.Context
import androidx.core.content.ContextCompat
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat.*

class AudioRecordingService : Service() {

    private var audioRecord: AudioRecord? = null
    private var isRecording = false
    private val SAMPLE_RATE = 16000
    private val CHANNEL = AudioFormat.CHANNEL_IN_MONO
    private val ENCODING = AudioFormat.ENCODING_PCM_16BIT
    private val TAG = "AudioRecordingService"

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(1, createNotification())
        startRecording()
        return START_STICKY
    }

    private fun startRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED
        ) {
            val bufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)

            if (bufferSize == AudioRecord.ERROR_BAD_VALUE || bufferSize == AudioRecord.ERROR) {
                Log.e(TAG, "Invalid buffer size for AudioRecord.")
                stopSelf() // Stop service if buffer size is invalid
                return
            }

            try {
                audioRecord = AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE,
                    CHANNEL,
                    ENCODING,
                    bufferSize
                )

                if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                    Log.e(TAG, "AudioRecord failed to initialize.")
                    stopSelf() // Stop service if AudioRecord failed to initialize
                    return
                }

                audioRecord?.startRecording()
                isRecording = true
                Log.d(TAG, "Recording started.")

                Thread {
                    val audioBuffer = ByteArray(bufferSize)
                    while (isRecording) {
                        val readResult = audioRecord?.read(audioBuffer, 0, audioBuffer.size)
                        if (readResult != null && readResult > 0) {
                            // Process audio data here (audioBuffer contains 'readResult' bytes of data)
                        } else if (readResult == AudioRecord.ERROR_INVALID_OPERATION ||
                            readResult == AudioRecord.ERROR_BAD_VALUE ||
                            readResult == AudioRecord.ERROR_DEAD_OBJECT ||
                            readResult == AudioRecord.ERROR) {
                            Log.e(TAG, "Error reading from AudioRecord: $readResult")
                            isRecording = false // Stop recording on error
                        }
                        // Add a small delay if necessary, e.g., if processing is very light
                        // Thread.sleep(10)
                    }
                    Log.d(TAG, "Recording thread finished.")
                }.start()

            } catch (e: SecurityException) {
                Log.e(TAG, "SecurityException: Missing RECORD_AUDIO permission. $e")
                // This catch block handles the specific error the linter is warning about.
                // Even with the checkSelfPermission above, a SecurityException could theoretically
                // occur in rare edge cases (e.g., permission revoked between check and use).
                stopSelf() // Stop the service as it cannot record.
            } catch (e: Exception) {
                Log.e(TAG, "SecurityException: Missing RECORD_AUDIO permission. $e")
                // This catch block handles the specific error the linter is warning about.
                // Even with the checkSelfPermission above, a SecurityException could theoretically
                // occur in rare edge cases (e.g., permission revoked between check and use).
                stopSelf() // Stop the service as it cannot record.
            } catch (e: Exception) {
                Log.e(TAG, "Exception during AudioRecord setup or recording: $e")
                stopSelf() // Stop service on other unexpected errors
            }
        } else {
            Log.w(TAG, "RECORD_AUDIO permission not granted. Cannot start recording.")
            // Optionally, update a notification to inform the user
            stopSelf() // Stop the service as it cannot record.
        }
    }

    private fun stopRecording() {
        if (isRecording) {
            isRecording = false // Signal the recording thread to stop
            try {
                audioRecord?.let {
                    if (it.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                        it.stop()
                    }
                    if (it.state == AudioRecord.STATE_INITIALIZED) {
                        it.release()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Exception while stopping AudioRecord: $e")
            } finally {
                audioRecord = null
                Log.d(TAG, "Recording stopped and resources released.")
            }
        }
    }

    override fun onDestroy() {
        stopRecording()
        super.onDestroy()
        Log.d(TAG, "AudioRecordingService destroyed.")
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotification(): Notification {
        val channelId = "AudioServiceChannel"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "Audio Background Service",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }

        val builder = NotificationCompat.Builder(this, channelId)
            .setContentTitle("Recording Audio")
            .setContentText("Recording in background...")
            .setSmallIcon(R.drawable.ic_mic)  // Replace with your app's icon

        return builder.build()
    }
}
