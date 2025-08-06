package com.example.emergencyaudiodetection

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.*
import android.telephony.SmsManager
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.example.emergencyaudiodetection.contact.ManageContactsActivity
import com.google.android.gms.location.LocationServices
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.ArrayDeque
import be.tarsos.dsp.AudioEvent
import be.tarsos.dsp.AudioProcessor
import be.tarsos.dsp.mfcc.MFCC
import be.tarsos.dsp.util.fft.FFT
import kotlin.math.log10
import kotlin.math.pow


class MainActivity : AppCompatActivity() {
    private lateinit var recordToggleButton: Button
    private lateinit var manageContactsButton: Button
    private var isRecording = false
    private var tflite: Interpreter? = null
    private var alertDialog: android.app.AlertDialog? = null
    private var autoSendAlertHandler: Handler? = null
    private var autoSendAlertRunnable: Runnable? = null
    private val confidenceWindow = ArrayDeque<Float>()
    private val smoothingWindowSize = 7
    private val coroutineScope = CoroutineScope(Dispatchers.IO)

    companion object {
        private const val SAMPLE_RATE = 16000
        private const val DURATION_SECONDS = 2
        private const val INPUT_LENGTH = SAMPLE_RATE * DURATION_SECONDS // 32000 samples
        private const val CONFIDENCE_THRESHOLD = 0.9f
        private const val PERMISSION_REQUEST_CODE = 100
        private const val AUTO_SEND_DELAY_MS = 10000L // 10 seconds
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        recordToggleButton = findViewById(R.id.recordButton)
        manageContactsButton = findViewById(R.id.manageContactsButton)

        recordToggleButton.setOnClickListener {
            if (!isRecording) {
                startRecording()
            } else {
                stopRecording()
            }
        }

        manageContactsButton.setOnClickListener {
            startActivity(Intent(this, ManageContactsActivity::class.java))
        }

        if (!hasAllPermissions()) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.RECORD_AUDIO,
                    Manifest.permission.SEND_SMS,
                    Manifest.permission.ACCESS_FINE_LOCATION
                ),
                PERMISSION_REQUEST_CODE
            )
        }

        try {
            tflite = Interpreter(loadModelFile("cnn_audio_classifier_approach (1).tflite"))
            // Log the input tensor shape
            // E.g., [15600] or [1, 15600]
            Log.d("ClassifierInputShape", tflite!!.getInputTensor(0).shape().joinToString())
            Log.d("ClassifierOutputShape", tflite!!.getOutputTensor(0).shape().joinToString())


        } catch (e: IOException) {
            Log.e("ModelInit", "Failed to load model", e)
            Toast.makeText(this, "Model loading error!", Toast.LENGTH_LONG).show()
        }
    }

    private fun hasAllPermissions(): Boolean {
        return ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) == PackageManager.PERMISSION_GRANTED &&
                ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED
    }
    private fun extractMelSpectrogram(
        audioBuffer: FloatArray,
        sampleRate: Int = 16000,
        nMels: Int = 40,
        nFFT: Int = 512,
        hopLength: Int = 256
    ): Array<Array<Array<FloatArray>>> {

        val fft = FFT(nFFT)
        val windowSize = nFFT
        val stepSize = hopLength
        val totalFrames = (audioBuffer.size - windowSize) / stepSize

        val powerSpectrogram = Array(totalFrames) { FloatArray(nFFT / 2) }

        for (i in 0 until totalFrames) {
            val start = i * stepSize
            val frame = FloatArray(windowSize) { j ->
                if (start + j < audioBuffer.size) audioBuffer[start + j] else 0f
            }

            // Prepare real + imaginary parts for FFT
            val fftBuffer = FloatArray(nFFT * 2)
            for (j in frame.indices) {
                fftBuffer[2 * j] = frame[j] // real
                fftBuffer[2 * j + 1] = 0f    // imaginary
            }

            fft.forwardTransform(fftBuffer)
            fft.modulus(fftBuffer, powerSpectrogram[i])
        }

        // Build mel filterbank
        val melFilterBank = createMelFilterBank(nFFT / 2, sampleRate, nMels)
        val melSpectrogram = Array(nMels) { FloatArray(totalFrames) }

        for (t in 0 until totalFrames) {
            for (m in 0 until nMels) {
                var melEnergy = 0f
                for (k in melFilterBank[m].indices) {
                    melEnergy += melFilterBank[m][k] * powerSpectrogram[t][k]
                }
                melSpectrogram[m][t] = log10(melEnergy + 1e-6f)
            }
        }

        // Pad or crop to 63 frames
        val paddedMel = Array(nMels) { row ->
            FloatArray(63) { col ->
                if (col < melSpectrogram[0].size) melSpectrogram[row][col] else 0f
            }
        }

        return arrayOf(Array(nMels) { i -> Array(63) { j -> floatArrayOf(paddedMel[i][j]) } })
    }

    private fun createMelFilterBank(
        nFftBins: Int,
        sampleRate: Int,
        nMels: Int
    ): Array<FloatArray> {

        fun hzToMel(hz: Float): Float = (2595 * log10(1 + hz / 700))
        fun melToHz(mel: Float): Float = 700f * (10f.pow(mel / 2595f) - 1f)


        val melMin = hzToMel(0f)
        val melMax = hzToMel(sampleRate / 2f)
        val melPoints = FloatArray(nMels + 2) { i ->
            melToHz(melMin + (i.toFloat() / (nMels + 1)) * (melMax - melMin))
        }

        val fftFrequencies = FloatArray(nFftBins) { i -> i * sampleRate.toFloat() / (nFftBins * 2) }

        val filterBank = Array(nMels) { FloatArray(nFftBins) }

        for (m in 1 until melPoints.size - 1) {
            val fLeft = melPoints[m - 1]
            val fCenter = melPoints[m]
            val fRight = melPoints[m + 1]

            for (k in 0 until nFftBins) {
                val freq = fftFrequencies[k]
                val weight = when {
                    freq < fLeft -> 0f
                    freq < fCenter -> (freq - fLeft) / (fCenter - fLeft)
                    freq < fRight -> (fRight - freq) / (fRight - fCenter)
                    else -> 0f
                }
                filterBank[m - 1][k] = weight
            }
        }

        return filterBank
    }


    private fun loadModelFile(modelName: String): MappedByteBuffer {
        assets.openFd(modelName).apply {
            return FileInputStream(fileDescriptor).channel.map(
                FileChannel.MapMode.READ_ONLY,
                startOffset,
                declaredLength
            )
        }
    }

    private fun startRecording() {
        // Permission check (can be kept as is)
        if (!hasAllPermissions()) {
            Toast.makeText(this, "Missing permissions.", Toast.LENGTH_LONG).show()
            return
        }

        if (tflite == null) {
            Toast.makeText(this, "Model not loaded!", Toast.LENGTH_LONG).show()
            return
        }

        // Explicit check before AudioRecord instantiation (add this)
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), PERMISSION_REQUEST_CODE)
            Toast.makeText(this, "Audio recording permission required", Toast.LENGTH_LONG).show()
            return
        }

        val bufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        // Wrap AudioRecord instantiation in try-catch to handle SecurityException:
        val audioRecord: AudioRecord
        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )
        } catch (se: SecurityException) {
            Log.e("AudioRecord", "Permission denied for AudioRecord", se)
            Toast.makeText(this, "Audio recording permission denied", Toast.LENGTH_LONG).show()
            return
        }

        val shortBuffer = ShortArray(INPUT_LENGTH) // 2 second buffer at 16kHz

        audioRecord.startRecording()
        isRecording = true
        recordToggleButton.text = getString(R.string.stop_listening)
        recordToggleButton.setBackgroundResource(R.drawable.btn_shape_disabled)
        Toast.makeText(this, "Recording started", Toast.LENGTH_SHORT).show()

        coroutineScope.launch {
            while (isRecording) {
                val readSamples = audioRecord.read(shortBuffer, 0, INPUT_LENGTH)
                if (readSamples == INPUT_LENGTH) {
                    val floatBuffer = FloatArray(INPUT_LENGTH) { i -> shortBuffer[i] / 32767.0f }

                    // Wrap floatBuffer into a 2D array for tflite input
                    val inputTensor = extractMelSpectrogram(floatBuffer)
                    val output = Array(1) { FloatArray(1) }

                    try {
                        tflite?.run(inputTensor, output)
                        val confidence = output[0][0]
                        confidenceWindow.addLast(confidence)
                        if (confidenceWindow.size > smoothingWindowSize) {
                            confidenceWindow.removeFirst()
                        }
                        val avgConfidence = confidenceWindow.average().toFloat()
                        Log.d("InferenceOutput", "Smoothed Confidence: $avgConfidence")
                        if (avgConfidence > CONFIDENCE_THRESHOLD) {
                            withContext(Dispatchers.Main) {
                                Toast.makeText(applicationContext, "🚨 Emergency Sound Detected!", Toast.LENGTH_SHORT).show()
                                if (alertDialog?.isShowing != true) {
                                    triggerAlert()
                                }
                            }
                        }
                    } catch (e: Exception) {
                        Log.e("InferenceError", "Failed to run inference", e)
                    }
                }
            }
            audioRecord.stop()
            audioRecord.release()
        }
    }


    private fun stopRecording() {
        isRecording = false
        recordToggleButton.text = getString(R.string.start_listening)
        recordToggleButton.setBackgroundResource(R.drawable.btn_shape_enabled)
        Toast.makeText(this, "Recording stopped", Toast.LENGTH_SHORT).show()
        dismissAlertDialog()
    }

    private fun triggerAlert() {
        alertDialog = android.app.AlertDialog.Builder(this)
            .setTitle("🚨 Emergency Detected")
            .setMessage("Emergency detected! Send alert to your contacts?")
            .setCancelable(false)
            .setPositiveButton("Yes") { _, _ -> sendAlert() }
            .setNegativeButton("No") { _, _ ->
                Toast.makeText(this, "Alert cancelled.", Toast.LENGTH_SHORT).show()
            }
            .create()

        alertDialog?.show()

        // Auto-send if no response
        autoSendAlertHandler = Handler(Looper.getMainLooper())
        autoSendAlertRunnable = Runnable {
            if (alertDialog?.isShowing == true) {
                alertDialog?.dismiss()
                sendAlert()
            }
        }
        autoSendAlertHandler?.postDelayed(autoSendAlertRunnable!!, AUTO_SEND_DELAY_MS)
    }

    private fun dismissAlertDialog() {
        autoSendAlertHandler?.removeCallbacks(autoSendAlertRunnable ?: return)
        alertDialog?.dismiss()
    }


    private fun sendAlert() {
        Log.d("SMS", "sendAlert() called")

        // Debug contacts
        debugContacts()

        // Test SMS permission
        if (!testSMSPermission()) {
            Toast.makeText(this, "SMS permission required", Toast.LENGTH_SHORT).show()
            return
        }

        val sharedPreferences = getSharedPreferences("EmergencyContacts", MODE_PRIVATE)
        val contactSet = HashSet(sharedPreferences.getStringSet("contacts", emptySet()) ?: emptySet())

        Log.d("ContactsDebug", "Loaded contacts: $contactSet")

        if (contactSet.isEmpty()) {
            Toast.makeText(this, "No emergency contacts saved. Please add contacts first.", Toast.LENGTH_LONG).show()
            return
        }

        // Get location and send SMS
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            val fusedLocationClient = LocationServices.getFusedLocationProviderClient(this)
            fusedLocationClient.lastLocation.addOnSuccessListener { location ->
                val locationText = location?.let {
                    "https://maps.google.com/?q=${it.latitude},${it.longitude}"
                } ?: "Location unavailable"

                Log.d("Location", "Location: $locationText")
                sendSmsToContacts(contactSet, locationText)
            }.addOnFailureListener { exception ->
                Log.e("Location", "Failed to get location", exception)
                sendSmsToContacts(contactSet, "Location unavailable")
            }
        } else {
            Log.w("Location", "Location permission not granted")
            sendSmsToContacts(contactSet, "Location unavailable")
        }
    }

    private fun sendSmsToContacts(contacts: Set<String>, location: String) {
        Log.d("SMS", "Attempting to send SMS to ${contacts.size} contacts")

        if (contacts.isEmpty()) {
            Toast.makeText(this, "No emergency contacts found", Toast.LENGTH_SHORT).show()
            return
        }

        // Check SMS permission
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "SMS permission not granted", Toast.LENGTH_SHORT).show()
            return
        }

        val smsManager: SmsManager = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            getSystemService(SmsManager::class.java)
        } else {
            SmsManager.getDefault()
        }

        val message = "🚨 EMERGENCY ALERT!\nI may be in danger and need help.\nLocation: $location\nPlease contact me or authorities immediately."

        var successCount = 0
        var failureCount = 0

        contacts.forEach { contact ->
            try {
                Log.d("SMS", "Sending SMS to: $contact")

                // Split long messages if needed
                val parts = smsManager.divideMessage(message)
                if (parts.size == 1) {
                    smsManager.sendTextMessage(contact, null, message, null, null)
                } else {
                    smsManager.sendMultipartTextMessage(contact, null, parts, null, null)
                }

                successCount++
                Log.d("SMS", "SMS sent successfully to: $contact")

            } catch (e: Exception) {
                failureCount++
                Log.e("SMS", "Failed to send SMS to $contact: ${e.message}", e)
            }
        }

        runOnUiThread {
            if (successCount > 0) {
                Toast.makeText(this, "🚨 Emergency alerts sent to $successCount contact(s)!", Toast.LENGTH_LONG).show()
            }
            if (failureCount > 0) {
                Toast.makeText(this, "Failed to send to $failureCount contact(s)", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun debugContacts() {
        val sharedPreferences = getSharedPreferences("EmergencyContacts", MODE_PRIVATE)
        val contactSet = HashSet(sharedPreferences.getStringSet("contacts", emptySet()) ?: emptySet())

        Log.d("ContactsDebug", "Total contacts: ${contactSet.size}")
        contactSet.forEachIndexed { index, contact ->
            Log.d("ContactsDebug", "Contact $index: '$contact'")
        }
    }

    private fun testSMSPermission(): Boolean {
        val hasPermission = ActivityCompat.checkSelfPermission(this, Manifest.permission.SEND_SMS) == PackageManager.PERMISSION_GRANTED
        Log.d("SMS", "SMS Permission granted: $hasPermission")
        return hasPermission
    }


    override fun onDestroy() {
        super.onDestroy()
        coroutineScope.cancel()
        tflite?.close()
        autoSendAlertHandler?.removeCallbacks(autoSendAlertRunnable ?: return)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, results)
        if (requestCode == PERMISSION_REQUEST_CODE) {
            if (!hasAllPermissions()) {
                Toast.makeText(this, "All permissions are required for emergency detection.", Toast.LENGTH_LONG).show()
            }
        }
    }
}

